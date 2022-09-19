# OAVNN

<img src="teaser.png" alt="Teaser image for project." width="640">

<h1>Abstract</h1>
Equivariant neural networks have been adopted in a variety of 3D learning areas. In this paper, we identify and address a fundamental problem of equivariant networks: their ambiguity to symmetries. For a left-right symmetric input, like an airplane, these networks cannot complete symmetry-dependent tasks like segmenting the object into its left and right sides. We tackle this problem by adding components that resolve symmetry ambiguities while preserving rotational equivariance. We present OAVNN: Orientation Aware Vector Neuron Network, an extension of the <a href=https://arxiv.org/abs/2104.12229> Vector Neuron Network</a>. OAVNN is a rotation equivariant network that is robust to planar symmetric inputs. Our network consists of three key components. First, we introduce an algorithm to calculate features for detecting symmetries. Second, we create an orientation-aware linear layer that is sensitive to symmetries. Finally, we construct an attention mechanism that relates directional information across points. We evaluate the network using left-right segmentation and find that the network quickly obtains accurate segmentations. We hope this work motivates investigations on the expressivity of equivariant networks to symmetric objects. 

<h1>Data</h1>
The code is structured to use any point cloud dataset of HDF5 type. Each file must only contain the x,y,z point locations (part labels are not necessary). We use the <a href=https://shapenet.org/> Shapenet dataset</a> for our experiments.

<h1>Running the Model</h1>

<ol>
  <li>Set up the conda environment using the provided environment.yml file.</li>
  <li>Train/Test the model using the following command:
    
        python run_models.py --exp_name NAME --model MODEL --rot ROT --class_choice CLASS --data_path PATH --num_points NUM
    
  - NAME is the name of the experiment. 
  - MODEL specifies which model to use and is one of the following: dgcnn, vnn, complex_only, shell_only, or oavnn. 
  - ROT specifies how the data is transformed, and is one of the following: aligned (aligned data), z (rotated only about vertical axis), so3 (full rotation). We present results using full so3 rotation. 
  - CLASS specifies the object class. We present results for aero, cap, and chair. 
  - PATH is a path from root to the data directory. Data directory should contain a subdirectory called "data_hdf5" with the HDF5 files. 
  - NUM is the number of points per example. We use 128 points per example.
    
  </li>
</ol>
