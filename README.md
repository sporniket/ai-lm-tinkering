# «Artificial Intelligence»/«Language Models» tinkering

The main goal is to build a Language Model that fit the following constraints : 

* Hardware requirements
  * Only need a CPU (no GPU/TPU/whatever)
  * Fit a RAM budget. There will be several profiles :
    * _XS_ : assuming a computer with at least 4 GiB RAM, the inference service with the model loaded in memory **MUST** leave at least 2 GiB of RAM available.
    * _S_ : assuming a computer with at least 8 GiB RAM, the inference service with the model loaded in memory **MUST** leave at least 4 GiB of RAM available.
    * _M_ : assuming a computer with at least 16 GiB RAM, the inference service with the model loaded in memory **MUST** leave at least 8 GiB.
    * _L_ : assuming a computer with at least 32 GiB RAM, the inference service with the model loaded in memory **MUST** leave at least 8 GiB.
    * **Nota Bene** : the inference service and the model **MAY** use tricks to stay within the memory budget while actually requiring more RAM than what is available (slicing, etc...)
* Performance requirement
  * Token generation **MUST** be fast enough. With a tolerance of 10% :
    * Pedagogical engines and models **SHOULD** need at most 5 seconds to generate a token.
    * Experimental engines and models **SHOULD** generate at least 100 tokens per minute.
    * Consolidated engines and models **SHOULD** generate at least 250 tokens per minute
