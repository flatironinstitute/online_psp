# online_psp
 The module `online_psp` offers a collection of computationally efficient algorithms for online subspace learning and principal component analysis, described in more detail in the
 accompanying paper [6].  There is an [analogous MATLAB package](https://github.com/simonsfoundation/online-psp-matlab) with a subset of the provided functionality.

## Installation 
The module can be installed by adding the parent directory
of online_psp to the Python path 
and compiling the Cython portion by running

```bash
python setup.py build_ext --inplace
```

## Dependencies
scikit-learn, matplotlib, numpy, pylab, cython




## Getting started
The `online_psp` module contains implementations of four different algorithms
for streaming principal subspace projection:

- Incremental PCA (IPCA) [3]
- Candid, Covariance-free IPCA [4]
- Similarity Matching (SM) [1]
- Fast Similarity Matching (FSM) [6]

The implementations of these may be found in the `online_psp` folder,
with accompanying demos in the `demo` folder, e.g., `online_psp/fast_similarity_matching.py` and `demo/demo_fast_similarity_matching.py`

The examples from the accompanying paper [6] are found in the `ex` directory.  These can take a long time to run.


### Documentation

```python
print('hello world')
```

### Options


## Data acknowledgments
The following example datasets have been included in this package.

- Yale face dataset, originally from the Yale vision group but obtained in processed form elsewhere 
(http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html)
- ORL database of faces from ATT Laboratories, Cambridge (http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)
- MNIST digits [5]

## References
[1] Pehlevan, C., Sengupta, A. and Chklovskii, D.B. "Why do similarity matching objectives lead to Hebbian/anti-Hebbian networks?." Neural computation 30, no. 1 (2018): 84-124.

[2] Cardot, H, and Degras, D. "Online Principal Component Analysis in High Dimension: Which Algorithm to Choose?." arXiv preprint arXiv:1511.03688 (2015).

[3] Arora, R., Cotter, A., Livescu, K. and Srebro, N., 2012, October. Stochastic optimization for PCA and PLS. In Communication, Control, and Computing (Allerton), 2012 50th Annual Allerton Conference on (pp. 861-868). IEEE.

[4] Weng, J., Zhang, Y. and Hwang, W.S., 2003. Candid covariance-free incremental principal component analysis. IEEE Transactions on Pattern Analysis and Machine Intelligence, 25(8), pp.1034-1040.

[5] LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P., 1998 “Gradient-based learning 
applied  to  document  recognition, Proceedings  of  the  IEEE,  vol.  86, no. 11, pp. 2278–2324.

[6] A forthcoming document to accompany this software.

## License
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
