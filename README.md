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

## Presence of LLM-generated content

The folder 00-ai-assisted-research will contains transcripts of chat with LLMs (Gemini, Claude, etc...), including sample code. Mostly in a Q&A mode, e.g. "explain me tokenization, with standard C, no libs". The content of that folder will presumably contains basic/public knowledge, and thus will be published under the Creative Common CC0 licence (kind of public domain-ish licence)

The chats are usually conducted in French and the transcripts will be copy/pasted as is. Each turn is entitled with a level 1 heading (e.g. "Tokenization"). When the prompt is basic/obvious (e.g. "teach me about tokenization"), it is omitted ; otherwise, the transcript of the prompt is put in a blockquote before the answer (e.g. "I know that tokenization consists in (...), but why is it that (...), I think I can do (...) ?")

