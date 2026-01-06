# UALR_2023_Full_Run
This was a full run of the 2023 USSC Data that does a breakdown of the features to the top 50 Principal Components.  

On the Lambda host (outside Docker)
Do this right after you SSH into the instance.

1. Create a workspace directory
mkdir -p ~/workspace
chmod 775 ~/workspace
cd ~/workspace
Put your files here, e.g.:
•	opafy23nid.parquet
•	master_analysis.py
•	drivers_by_offense_grouped_clean_fy2023_cpu.py
•	run_mapper_fy2023.py
•	build_sankey_2023.py (if you have it)
You can upload via SCP or Jupyter later, but this is the host-side mount point.

2. Pull the RAPIDS image (the one we know works)
sudo docker pull rapidsai/rapidsai:23.08a-cuda12.0.1-py3.10

3. Start the RAPIDS container with GPU + port 8888 exposed
This is the correct docker run that gives you:
•	GPU access
•	Shared /workspace
•	Port 8888 mapped so Jupyter is reachable
sudo docker run -d \
  --gpus all \
  --shm-size=32g \
  -v ~/workspace:/workspace \
  -w /workspace \
  -p 8888:8888 \
  --name rapids-research \
  rapidsai/rapidsai:23.08a-cuda12.0.1-py3.10 \
  sleep infinity
Check it’s running and sees the GPU:
sudo docker exec -it rapids-research nvidia-smi
If you see the H100 or whatever GPU you chose for you instance, you’re good.

3. Inside the RAPIDS container
Now hop into the container:
sudo docker exec -it rapids-research bash
Prompt should look like:
(base) rapids@xxxxxxxx:/workspace$
Everything from now on is inside the container.

4. Install JupyterLab + a few extra Python libs
Important: RAPIDS image already has a working xgboost & numpy stack.
We do not want to overwrite that → so do NOT pip install xgboost or numpy.

Run:
pip install --upgrade pip
pip install jupyterlab ipywidgets python-igraph plotly seaborn shap kmapper

Note: I deliberately left out xgboost from that list to avoid the 1.7.4 vs 1.7.6 mismatch I hit earlier.

5. Quick sanity check of key libraries
Still inside container:
python - << 'EOF'
import pandas as pd
import xgboost
import shap
import kmapper as km
import cudf, cupy as cp

print("pandas:", pd.__version__)
print("xgboost:", xgboost.__version__)
print("shap:", shap.__version__)
print("kmapper:", km.__version__)
print("cudf:", cudf.__version__)
print("GPU test:", cp.arange(5))
EOF
If that runs without error: environment is good.

6. Start JupyterLab inside the container
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''
Leave that process running.

7. Connect from another brower window:
 
In your browser, go to:
http://<your-lambda-public-ip>:8888/lab
(Same IP you use for SSH, e.g. 209.20.159.19.)
Note: you may have to make ports available to connect, or it may not see the lab.

You should see JupyterLab.
If you can’t reach it:
•	Double-check the container was started with -p 8888:8888 (step 3)
•	Make sure JupyterLab is running in that container (you’ll see logs in the SSH window)

8. Running your research correctly in Jupyter

Inside JupyterLab file browser, you’ll see /workspace with your files.
Key rule:
On Lambda, your parquet is at /workspace/opafy23nid.parquet
\
So in a notebook:

run the following:

!python master_analysis.py


from master_analysis import run_all  # if structured that way

input_path = "/workspace/opafy23nid.parquet"
output_folder = "fy2023_analysis_results"

run_all(input_path=input_path, out_dir=output_folder)
Or, if you want to run the script as-is:
%run master_analysis.py
Make sure inside that script you’ve changed any /content/... paths to /workspace/....
Then:
%run drivers_by_offense_grouped_clean_fy2023_cpu.py
# and later:
%run run_mapper_fy2023.py
# etc.
cuda12.0.1-py3.10 sleep infinity
2.	Inside container
o	sudo docker exec -it rapids-research bash
o	pip install jupyterlab ipywidgets python-igraph plotly seaborn shap kmapper
o	Run the Python sanity check
o	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''
3.	On your laptop
o	Open http://<lambda-ip>:8888/lab
o	Use /workspace/opafy23nid.parquet as your input path
o	%run master_analysis.py
o	%run drivers_by_offense_grouped_clean_fy2023_cpu.py
o	%run run_mapper_fy2023.py
If you want, I can also:
•	Clean up master_analysis.py for you so it:
o	uses /workspace paths
o	has a main() or run_all()
o	can be run via %run or python master_analysis.py
and then we’ll be fully ready to focus on interpreting the results, not wrestling the environment.

