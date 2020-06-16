# Thesis
Statistical Mechanics for monodisperse granular materials.
These codes use the LIGGGHTS software to generate a granular material distribution and realize experiments on it.
Additionally the voro++ library is used to calculate Voronoi cells of the distribution.
Thanks to professor William Oquendo for his support on post processing LIGGGHTS files.
The order in which codes should be run to recreate the local folder is as follows:

    1. Run the LIGGGHTS simulation
    
       mpirun -np 4 '/path_to_liggghts/LIGGGHTS-PUBLIC/src/lmp_auto' -i in.packing_dilute
       
       For a simulation using wall compression; this uses 4 cores and requires the .stl files located on the /meshes/ folder
       This will output dump files on /out and /out_vtk, containing info regarding particles and contacts. If you need visualization use pareview to import the .vtk files.
       
       <REQUIREMENTS>: An installation of LIGGGHTS that can dump .gz files.
       TODO: Implement periodic boundaries for LIGGGHTS
       
    2. Run the postprocessing for dump files
    
       python3 postproc.py
        
       This will output a bunch of files on the /post folder.
       Specifically:
        - Cundall parameter for keeping track of force equilibrium
        - Packing fraction for keeping track of volume equilibrium
        - Mean coordination number for various statistical distributions
        - Histogram: Pressure vs Number of Particles
        - Histogram: Voronoi cell volume vs Number of cells
        - Array containing Pressure for every particle
        - Array containing volume for every Voronoi cell
        - .pov files of particles and voronoi cells for visualization using povray. Notice these files are specially large, so manage your space accordingly. (That is, don't post process an enormous amount of files)
        
        <REQUIREMENTS>: python3 and the libraries os, sys, numpy, glob, colorama. An installation of voro++ to calculate voronoi cells of the distribution.
        
    3. Run the plot generator for the postprocessed files
       
       python3 postHisto.py
       
       This will generate plots in the /plots folder for:
       - Equilibrium parameters.
       - PDF for Voronoi cell volume and Pressure (you need to choose an equilibrium time manually)
       - Histograms for Voronoi cell volume and Pressure
       
       <REQUIREMENTs>: python3's matplot and scipy libraries
       
    4. Run the bash jobs to generate images of the packing
    
        ./voro_img.job
        
        This will generate images for the packing in each timestep using the povray script import.pov.
        
        <REQUIREMENTS>: povray version 3.6 or higher
        
    5. Run the /anim bash jobs
    
        ./sorf_files.job
        
        to generate a "images.txt" file containing the order in which images should be rendered to create a video
        
        ./images_to_video_ffmpeg.job <filename>.mp4
        
        to generate a video using the images.
        
        <REQUIREMENTS>: ffmpeg
