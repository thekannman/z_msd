//Copyright (c) 2015 Zachary Kann
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

// ---
// Author: Zachary Kann

//TODO(Zak): Reintroduce depth profile as in earlier version of program.

#include <complex>
#include "xdrfile_trr.h"
#include "boost/program_options.hpp"
#include "z_sim_params.hpp"
#include "z_vec.hpp"
#include "z_constants.hpp"
#include "z_conversions.hpp"
#include "z_molecule.hpp"
#include "z_subsystem_group.hpp"
#include "z_histogram.hpp"
#include "z_gromacs.hpp"

namespace po = boost::program_options;
// Units are nm, ps.

int main (int argc, char *argv[]) {
  SimParams params;

  po::options_description desc("Options");
  desc.add_options()
    ("help,h",  "Print help messages")
    ("group,g", po::value<std::string>()->required(),
     "Group to use for msd calculation.")
    ("index,n", po::value<std::string>()->default_value("index.ndx"),
     ".ndx file containing atomic indices for groups")
    ("gro", po::value<std::string>()->default_value("conf.gro"),
     ".gro file containing list of atoms/molecules")
    ("top", po::value<std::string>()->default_value("topol.top"),
     ".top file containing atomic/molecular properties")
    ("max_time,t", po::value<double>()->default_value(0.0),
     "Maximum simulation time to use in calculations");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    exit(EXIT_SUCCESS);
  }

  std::map<std::string, std::vector<int> > groups;
  groups = ReadNdx(vm["index"].as<std::string>());
  std::vector<Molecule> molecules = GenMolecules(vm["top"].as<std::string>(),
                                                 params);

  SystemGroup all_atoms(vm["gro"].as<std::string>(), molecules);

  SubsystemGroup *selected_group_pointer =
      SubsystemGroup::MakeSubsystemGroup(
          vm["group"].as<std::string>(),
          SelectGroup(groups, vm["group"].as<std::string>()), all_atoms);
  SubsystemGroup &selected_group = *selected_group_pointer;

  rvec *x_in = NULL;
  matrix box_mat;
  arma::rowvec box = arma::zeros<arma::rowvec>(DIMS);
  std::string xtc_filename = "prod.xtc";
  XDRFILE *xtc_file;
  params.ExtractTrajMetadata(strdup(xtc_filename.c_str()), (&x_in), box);
  xtc_file = xdrfile_open(strdup(xtc_filename.c_str()), "r");
  params.set_box(box);
  params.set_max_time(vm["max_time"].as<double>());

  int correlation_length = 3000;
  Histogram msd(correlation_length, params.dt());
  arma::cube old_positions =
      arma::zeros<arma::cube>(DIMS, selected_group.num_molecules(),
                              correlation_length);

  arma::rowvec dx;
  int st;
  float time, prec;
  int step = 0;
  for (step = 0; step < params.max_steps(); ++step) {
    int mod = step % correlation_length;
    if(read_xtc(xtc_file, params.num_atoms(), &st, &time, box_mat, x_in, &prec))
      break;

    params.set_box(box_mat);

    selected_group.set_positions(x_in);
    old_positions.slice(mod) = selected_group.com_positions();
    for (int i_step = step - 1; i_step > step - correlation_length; i_step--) {
      if(i_step < 0) continue;
      int old_mod = i_step % correlation_length;
      int step_diff = step - i_step;
      for (int i_mol = 0; i_mol < selected_group.num_molecules(); i_mol++) {
        FindDxNoShift(dx, old_positions.slice(mod).col(i_mol),
                      old_positions.slice(old_mod).col(i_mol), params.box());
        double r2 = arma::dot(dx,dx);
        msd.Add(step_diff, r2);
      }
    }
  }
  xdrfile_close(xtc_file);

  msd.Print("msd.txt", true);

}
 // main
