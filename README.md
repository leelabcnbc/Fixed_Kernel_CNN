# Quick start

The image stimuli used in the experiments can be found in `./data/stimuli/`. The main experiments were performed on the pattern stimuli, but some additional work mentioned in the discussion and described in the supplementary materials was done on the natural image data.

The neural responses are in `./data/neurons/`. A couple of example neurons for the pattern data are stored, for the purposes of the demo, and the full neural data for each monkey/stimuli set are saved accordingly, all as Numpy arrays.


The demo.ipynb file has a step-by-step tutorial of how to use PPR, CPPR, CMPR, FKCNN, GCNN on these neurons.


# 8K data
Please refer to 8K_FKCNN.ipynb for details on how to train FKCNN on 8K data (on real-world stimuli). Also I have implemented the FKCNN in Yimeng's folder, thesis-proposal-v2 branch FKCNN. If anything works unexpected please use my FKCNN implementation in Yimeng's github repo.

8K_PPR.ipynb contains how to train and compare PPR, CPPR and CMPR on 8K but the result is very bad, suggesting pursuit regression is not a good set of models to fit 8K data. 

## Citation
If you find this repo useful, please cite the following paper:
```
@misc{wu2020complex,
      title={Complex Sparse Code Priors Improve the Statistical Models of Neurons in Primate Primary Visual Cortex}, 
      author={Ziniu Wu and Harold Rockwell and Yimeng Zhang and Shiming Tang and Tai Sing Lee},
      year={2020},
      eprint={1911.08241},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM}
}
```
