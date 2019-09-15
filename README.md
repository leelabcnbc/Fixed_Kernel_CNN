# Projection Pursuit Regression
The full code for PPR, CPPR, CMPR model can be found in folder run_script.


# Fixed-Kernel CNN

I changed my implementation of CNN structure into Yimeng's stucture in jcompneuro, which is contained in the subdirectory tang_jcompneuro with some of my own modification. Please refer to Demo.ipynb to understand how to build and train FKCNN model on Tang's data. Make sure you set correct environment and data path before training. 


# 8K data
Please refer to 8K_FKCNN.ipynb for details on how to train FKCNN on 8K data. Also I have implemented the FKCNN in Yimeng's folder, thesis-proposal-v2 branch FKCNN. If anything works unexpected please use my FKCNN implementation in Yimeng's github repo.

8K_PPR.ipynb contains how to train and compare PPR, CPPR and CMPR on 8K but the result is very bad, suggesting pursuit regression is not a good set of models to fit 8K data. 