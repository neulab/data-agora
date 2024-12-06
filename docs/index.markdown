---
layout: default
---


## Motivation
{: .sys-img}
![Motivation of AgoraBench.](/assets/img/motivation.png)

Prior works on synthetic data generation have primarily focused on proposing better methods for generating high-quality data, which led to various experimental settings. For instance, Self-Instruct, Alpaca, WizardLM, and Orca varied in their choice of LMs for data generation, quantity of synthetic training data, base models used for training, and benchmarks for evaluating the model trained on the synthetic data.

<br>
As shown in the Figure above, this makes it difficult to directly compare how good different LMs are as a data generator. Specifically, while GPT-4o is 3.4 times expensive than GPT-4o, is it really worth the cost? On the other hand, as we have a lot of good high-performing LMs nowadays, should we really use proprietary LMs for data generation or would open-source LMs such as Llama 3 be a good alternative?

<br>
To answer these kind of questions, we need a more systematic approach to evaluate how good a LM is as a data generator. More specifically, we need a unified experimental setting where only the data generator varies and all the other components are fixed. In this work, we introduce AgoraBench, a benchmark that serves this purpose by provided 9 experimental settings.


## Data Generation Methods
{: .sys-img}
![Data Generation Methods covered in AgoraBench.](/assets/img/methods.png)

<br>
In AgoraBench, we cover the following data generation methods:
* <b>Instance Generation</b>: Similar to Self-Instruct, the data generator is conditioned on in-context demonstrations and generates new instances.
* <b>Response Generation</b>: Given a fixed set of instructions, the data generator generates responses for each instruction.
* <b>Quality Enhancement</b>: Given large amounts of low-quality data, the data generator enhances the quality of the data.


## Metrics
{: .sys-img}
![Performance Gap Recovered (PGR) metric used in AgoraBench.](/assets/img/metrics.png)

<br>
To quantify the quality of the generated data (i.e., data generation capability), we propose a new metric called **Performance Gap Recovered (PGR)**. On a high level, PGR measures how much a student model trained on the synthetic data improves over its base model compared to a reference model that shares the same base model. 

<br>
Specifically, we use Llama-3.1-8B as our base model and Llama-3.1-8B-Instruct as the reference model. This captures how much the synthetic data is able to improve the performance of the student model compared to Meta's post-training process for obtaining Llama-3.1-8B-Instruct with Llama-3.1-8B.


## AgoraBench Results
{: .sys-img}
![AgoraBench results.](/assets/img/agorabench_results.png)

<br>
We find that different models have distinct strengths and weaknesses in each data generation method. For instance, while GPT-4o excels in instance generation, Claude-3.5-Sonnet shows stronger performance in quality enhancement.


## Conclusion
For more information about our work, please check out our paper! Also, we plan to continually update our model based on your feedback! Feel free to reach out to us via email or twitter!

## Bibtex
<pre>
@misc{kim2024evaluating,
      title={Evaluating Language Models as Synthetic Data Generators}, 
      author={Seungone Kim and Juyoung Suk and Xiang Yue and Vijay Viswanathan and Seongyun Lee and Yizhong Wang and Kiril Gashteovski and Carolin Lawrence and Sean Welleck and Graham Neubig},
      year={2024},
      eprint={2412.03679},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.03679}, 
}
</pre>

------

{: .logos}
[![Logo of CMU](/assets/img/cmu.png)](https://www.lti.cs.cmu.edu/)


{: .center .acknowledgement}
This research was supported by the **NEC Student Research Fellowship Program**.
