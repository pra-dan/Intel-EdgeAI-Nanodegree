#!/usr/bin/env python
# coding: utf-8

# # Exercise: CPU and the Devcloud
# 
# Now that we've walked through the process of requesting a CPU on Intel's DevCloud and loading a model, you will have the opportunity to do this yourself with the addition of running inference on an image.
# 
# In this exercise, you will do the following:
# 1. Write a Python script to load a model and run inference 10 times on a CPU on Intel's DevCloud.
#     * Calculate the time it takes to load the model.
#     * Calculate the time it takes to run inference 10 times.
# 2. Write a shell script to submit a job to Intel's DevCloud.
# 3. Submit a job using `qsub` on the **IEI Tank-870** edge node with an **Intel Xeon E3 1268L v5**.
# 4. Run `liveQStat` to view the status of your submitted job.
# 5. Retrieve the results from your job.
# 6. View the results.
# 
# Click the **Exercise Overview** button below for a demonstration.

# <span class="graffiti-highlight graffiti-id_g9b3e7l-id_08m53df"><i></i><button>Exercise Overview</button></span>

# #### IMPORTANT: Set up paths so we can run Dev Cloud utilities
# You *must* run this every time you enter a Workspace session.

# In[43]:


get_ipython().run_line_magic('env', 'PATH=/opt/conda/bin:/opt/spark-2.4.3-bin-hadoop2.7/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/intel_devcloud_support')
import os
import sys
sys.path.insert(0, os.path.abspath('/opt/intel_devcloud_support'))
sys.path.insert(0, os.path.abspath('/opt/intel'))


# ## The Model
# 
# We will be using the `vehicle-license-plate-detection-barrier-0106` model for this exercise. Remember that to run a model on the CPU, we need to use `FP32` as the model precision.
# 
# The model has already been downloaded for you in the `/data/models/intel` directory on Intel's DevCloud.
# 
# We will be running inference on an image of a car. The path to the image is `/data/resources/car.png`

# # Step 1: Creating a Python Script
# 
# The first step is to create a Python script that you can use to load the model and perform inference. We'll use the `%%writefile` magic to create a Python file called `inference_cpu_model.py`. In the next cell, you will need to complete the `TODO` items for this Python script.
# 
# `TODO` items:
# 
# 1. Load the model
# 
# 2. Prepare the model for inference (create an input dictionary)
# 
# 3. Run inference 10 times in a loop
# 
# If you get stuck, you can click on the **Show Solution** button below for a walkthrough with the solution code.

# In[44]:


get_ipython().run_cell_magic('writefile', 'inference_cpu_model.py', '\nimport time\nimport numpy as np\nimport cv2\nfrom openvino.inference_engine import IENetwork\nfrom openvino.inference_engine import IECore\nimport argparse\n\ndef main(args):\n    model=args.model_path\n    model_weights=model+\'.bin\'\n    model_structure=model+\'.xml\'\n    \n    start=time.time()\n    \n    # TODO: Load the model\n    model=IENetwork(model_structure, model_weights)\n\n    core = IECore()\n    net = core.load_network(network=model, device_name=\'CPU\', num_requests=1)\n    \n    print(f"Time taken to load model = {time.time()-start} seconds")\n    \n    # Get the name of the input node\n    input_name=next(iter(model.inputs))\n    \n    # Reading and Preprocessing Image\n    input_img=cv2.imread(\'/data/resources/car.png\')\n    input_img=cv2.resize(input_img, (300,300), interpolation = cv2.INTER_AREA)\n    input_img=np.moveaxis(input_img, -1, 0)\n\n    # TODO: Prepare the model for inference (create input dict etc.)\n    input_dict={input_name:input_img}\n    \n    start=time.time()\n    for _ in range(10):\n        net.infer(input_dict)\n        \n    \n    print(f"Time Taken to run 10 inference on CPU is = {time.time()-start} seconds")\n\nif __name__==\'__main__\':\n    parser=argparse.ArgumentParser()\n    parser.add_argument(\'--model_path\', required=True)\n    \n    args=parser.parse_args() \n    main(args)')


# <span class="graffiti-highlight graffiti-id_6t269sv-id_2g8nwk3"><i></i><button>Show Solution</button></span>

# ## Step 2: Creating a Job Submission Script
# 
# To submit a job to the DevCloud, you'll need to create a shell script. Similar to the Python script above, we'll use the `%%writefile` magic command to create a shell script called `inference_cpu_model_job.sh`. In the next cell, you will need to complete the `TODO` items for this shell script.
# 
# `TODO` items:
# 1. Create a `MODELPATH` variable and assign it the value of the first argument that will be passed to the shell script
# 2. Call the Python script using the `MODELPATH` variable value as the command line argument
# 
# If you get stuck, you can click on the **Show Solution** button below for a walkthrough with the solution code.

# In[45]:


get_ipython().run_cell_magic('writefile', 'inference_cpu_model_job.sh', '#!/bin/bash\n\nexec 1>/output/stdout.log 2>/output/stderr.log\n\nmkdir -p /output\n\nMODELPATH=$1\n\n# Run the load model python script\npython3 inference_cpu_model.py  --model_path ${MODELPATH}\n\ncd /output\n\ntar zcvf output.tgz stdout.log stderr.log')


# <span class="graffiti-highlight graffiti-id_vc779df-id_z9ijl86"><i></i><button>Show Solution</button></span>

# ## Step 3: Submitting a Job to Intel's DevCloud
# 
# In the next cell, you will write your `!qsub` command to submit your job to Intel's DevCloud to load your model on the **Intel Xeon E3 1268L v5** CPU and run inference.
# 
# Your `!qsub` command should take the following flags and arguments:
# 1. The first argument should be the shell script filename
# 2. `-d` flag - This argument should be `.`
# 3. `-l` flag - This argument should request a **Tank-870** node using an **Intel Xeon E3 1268L v5** CPU. The default quantity is 1, so the **1** after `nodes` is optional.
# To get the queue label for this CPU, you can go to [this link](https://devcloud.intel.com/edge/get_started/devcloud/)
# 4. `-F` flag - This argument should be the full path to the model. As a reminder, the model is located in `/data/models/intel`.
# 
# **Note**: There is an optional flag, `-N`, you may see in a few exercises. This is an argument that only works on Intel's DevCloud that allows you to name your job submission. This argument doesn't work in Udacity's workspace integration with Intel's DevCloud.
# 
# If you get stuck, you can click on the **Show Solution** button below for a walkthrough with the solution code.

# In[46]:


job_id_core = get_ipython().getoutput('qsub inference_cpu_model_job.sh -d . -l nodes=1:tank-870:e3-1268l-v5 -F "/data/models/intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106" -N store_core')
print(job_id_core[0])


# <span class="graffiti-highlight graffiti-id_qnrfru0-id_7ofr7nk"><i></i><button>Show Solution</button></span>

# ## Step 4: Running liveQStat
# 
# Running the `liveQStat` function, we can see the live status of our job. Running the this function will lock the cell and poll the job status 10 times. The cell is locked until this finishes polling 10 times or you can interrupt the kernel to stop it by pressing the stop button at the top: ![stop button](assets/interrupt_kernel.png)
# 
# * `Q` status means our job is currently awaiting an available node
# * `R` status means our job is currently running on the requested node
# 
# **Note**: In the demonstration, it is pointed out that `W` status means your job is done. This is no longer accurate. Once a job has finished running, it will no longer show in the list when running the `liveQStat` function.
# 
# Click the **Running liveQStat** button below for a demonstration.

# <span class="graffiti-highlight graffiti-id_9xnofi7-id_m9v5xi8"><i></i><button>Running liveQStat</button></span>

# In[47]:


import liveQStat
liveQStat.liveQStat()


# ## Step 5: Retrieving Output Files
# 
# In this step, we'll be using the `getResults` function to retrieve our job's results. This function takes a few arguments.
# 
# 1. `job id` - This value is stored in the `job_id_core` variable we created during **Step 3**. Remember that this value is an array with a single string, so we access the string value using `job_id_core[0]`.
# 2. `filename` - This value should match the filename of the compressed file we have in our `inference_cpu_model_job.sh` shell script.
# 3. `blocking` - This is an optional argument and is set to `False` by default. If this is set to `True`, the cell is locked while waiting for the results to come back. There is a status indicator showing the cell is waiting on results.
# 
# **Note**: The `getResults` function is unique to Udacity's workspace integration with Intel's DevCloud. When working on Intel's DevCloud environment, your job's results are automatically retrieved and placed in your working directory.
# 
# Click the **Retrieving Output Files** button below for a demonstration.

# <span class="graffiti-highlight graffiti-id_u14xt9e-id_pksums3"><i></i><button>Retrieving Output Files</button></span>

# In[48]:


import get_results
get_results.getResults(job_id_core[0], filename="output.tgz", blocking=True)


# ## Step 6: View the Outputs
# In this step, we unpack the compressed file using `!tar zxf` and read the contents of the log files by using the `!cat` command.
# 
# `stdout.log` should contain the printout of the print statement in our Python script.

# In[49]:


get_ipython().system('tar zxf output.tgz')


# In[50]:


get_ipython().system('cat stdout.log')


# In[51]:


get_ipython().system('cat stderr.log')

