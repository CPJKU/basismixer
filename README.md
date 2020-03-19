# basismixer
The Basis Mixer is an implementation of the [Basis Function Modeling framework for musical expression](http://www.carloscancinochacon.com/documents/online_extras/phd_thesis/basis_function_models.html). 

The basic idea in this framework is that structural properties of a musical piece (given as a score), which are believed to be relevant for performance decisions, can be modeled in a simple and uniform way via so-called basis functions: numeric features that capture specific aspects of a musical note and its surroundings. A predictive model of performance can then predict appropriate patterns for expressive performance dimensions such as tempo, timing, dynamics and articulation from these basis functions.

## Setup

0. **Optional** Create an environment for the dependencies

	```bash
	conda env create -f environment.yml
	```
	
	As of this writing (March 2020), it is recommended to install the Partitura package directly from the [GitHub repository](https://github.com/OFAI/partitura) instead of from the version in PyPI. This can be done by simply running

	```bash
	pip install git+https://github.com/OFAI/partitura.git@develop
	```

1. The easiest way to install the basismixer package is with pip.

	```bash
	git clone https://github.com/OFAI/basismixer.git
	pip install ./basismixer
	```

## Usage

Here is a quick guide for using this package. 

### Tutorial

We prepared a collection of Jupyter notebooks demonstrating this package, which can be found [here](https://github.com/mgrachten/basismixer-notebooks). These notebooks were presented during the tutorial [*Computational Modeling of Musical Expression: Perspectives, Datasets, Analysis and Generation*](https://ismir2019.ewi.tudelft.nl/?q=node/39) presented at the ISMIR 2019 conference in Delft, the Netherlands. You can test these notebooks [online](https://mybinder.org/v2/gh/mgrachten/basismixer-notebooks/master).


### Generating an Expressive Performance

To generate an expressive performance

```bash
./bin/render_piece [OPTIONS] score_file midi_file.mid
```

The score (`score_file`) can be in one of the formats supported by the [partitura package](https://partitura.readthedocs.io/en/latest/). 

This repository includes a small model trained on the Vienna 4x22 dataset. This model was prepared for the tutorial on Computational Models of Expressive Performance presented at the ISMIR 2019. **This is demonstration model and is not intended to represent the state-of-the-art.**

For more help see

```bash
./bin/render_piece -h
```

### Training a Model
See notebook [`03_predictive_models.ipynb`](https://github.com/mgrachten/basismixer-notebooks/blob/master/03_predictive_models.ipynb) in the tutorial.

## Citing this work

* C. E. Cancino-Chacón and M. Grachten (2016) The Basis Mixer: A Computational Romantic Pianist (2016) In *Late Breaking/Demo at the 17th International Society for Music Information Retrieval Conference (ISMIR 2016)*. New York, New York, USA [(pdf)](http://www.carloscancinochacon.com/documents/extended_abstracts/CancinoGrachten-ISMIR2016-LBD-ext-abstract.pdf)

	```bibtex
	@inproceedings{CancinoChacon:2016wi,
		Address = {New York, NY, USA},
		Author = {Cancino-Chac{\'o}n, Carlos Eduardo and Grachten, Maarten},
		Booktitle = {Proceedings of the Late Breaking/ Demo Session, 17th International Society for Music Information Retrieval Conference (ISMIR 2016)},
		Title = {{The Basis Mixer: A Computational Romantic Pianist}},
		Year = {2016}}
	```

* C. E. Cancino-Chacón, T. Gadermaier, G. Widmer and M. Grachten (2017) An Evaluation of Linear and Non-linear Models of Expressive Dynamics in Classical Piano and Symphonic Music. *Machine Learning* **109**, 887–909. [(link)](https://doi.org/10.1007/s10994-017-5631-y)

	```bibtex
	@article{CancinoChaconEtAl:2017,
		Author = {Cancino-Chac{\'o}n, Carlos Eduardo and Gadermaier, Thassilo and Widmer, Gerhard and Grachten, Maarten},
		Journal = {Machine Learning},
		Number = {6},
		Pages = {887--909},
		Title = {{An Evaluation of Linear and Non-linear Models of Expressive Dynamics in Classical Piano and Symphonic Music}},
		Volume = {106},
		doi = {10.1007/s10994-017-5631-y},
		Year = {2017}}
	```
	

* C. E. Cancino-Chacón (2018) *Computational Modeling of Expressive Music Performance with Linear and Non-linear Basis Function Models*. PhD thesis. Johannes Kepler University Linz, Austria [(pdf)](http://www.carloscancinochacon.com/documents/thesis/Cancino-JKU-2018.pdf)

	```bibtex

	@phdthesis{Cancino2018thesis,
		Address = {Austria},
		Author = {Cancino-Chac\'on, Carlos Eduardo},
		Month = {December},
		School = {Johannes Kepler University Linz},
		Title = {{Computational Modeling of Expressive Music Performance with Linear and Non-linear Basis Function Models}},
		Year = {2018}}
	```

## License

The code in this package is licensed under the GNU General Public License v3 (GPLv3). For details, please see the LICENSE file.

## Acknowledgements

This research has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme under grant agreement No. 670035 (project [*Con Espressione*](https://www.jku.at/en/institute-of-computational-perception/research/projects/con-espressione/)).
