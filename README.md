### Building

To build the non-linear elasticity solver and examples, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
The examples' executables will be created in subdirectories of `build/src/examples/`, and can be executed through
```bash
$ ./executable-name
```

### Heart mesh

The heart examples require `heart_mesh.msh` and `material_coordinates.txt` to be present
in the current working directory during runtime. 
These files can be generated in the `heart-mesh/` directory by creating and activating
a Python virtual environment, ensuring `gmsh` pip package is installed, and then running the
`heart-gen.py` script. 
If the files `heart_mesh.msh` and `material_coordinates.txt` are present in the `heart-mesh/`
directory during CMake configuration, they will be automatically copied to the 
`build/src/examples/heart/` directory. Otherwise, CMake will print a warning.
