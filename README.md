# vasp_utils 

This repo contains scripts for various VASP workflows, including job submission (with slurm) and post-analyzing (with python3). 



# Setup

1. clone the repos and add them to your PATH:     
   https://github.com/BinglunYin/vasp_utils    
   https://github.com/BinglunYin/slurm_utils    

1. pip install a package:     
   ```shell
   pip3 install --upgrade --user   git+https://github.com/BinglunYin/myalloy_package.git  
   ```

1. add a link to the python3 at:  
   `$HOME/opt/bin/python3`



# Usage

1. Calculate the reference state in the folder `y_full_relax`. This folder will serve as the basis for the following workflows.  

1. For example. If you would like to calculate the EOS of the reference state, run the following command at the level of `y_full_relax`:    
    ```shell
    yin_vasp_run_eos
    ```

    When all the jobs are done, run the command:    
    ```shell
    yin_vasp_plot_all  -eos 
    ```

   Then you will have the EOS result.   
   
1. All the workflows commands are with the name `yin_vasp_run_*`.
   


