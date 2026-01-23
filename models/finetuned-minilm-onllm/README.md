---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:96
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: What fairness or bias risks arise in machine learning systems,
    and what mitigation approaches are used?
  sentences:
  - I . INTRODUCTION I NDUSTRIAL Internet of Things ( IoT ) and data-driven techniques
    have been revolutionizing manufacturing by enabling computer networks to gather
    the huge amount of data from connected machines and turn the big machinery data
    into actionable information [ 1 ] , [ 2 ] , [ 3 ] . As a key component in modern
    manufacturing system , machine health monitoring has fully embraced the big data
    revolution . Compared to top-down modeling provided by the traditional physics-based
    models [ 4 ] , [ 5 ] , [ 6 ] , data-driven machine health monitoring systems offer
    a new paradigm of bottom-up solution for detection of faults after the occurrence
    of certain failures ( diagnosis ) and predictions of the future working conditions
    and the remaining useful life ( prognosis ) [ 1 ] , [ 7 ] . As we know , the complexity
    and noisy working condition hinder the construction of physical models . And most
    of these physicsbased models are unable to be updated with on-line measured data
    , which limits their effectiveness and flexibility . On the other hand , with
    significant development of sensors , sensor networks and computing systems , data-driven
    machine health monitoring models have become more and more attractive . To extract
    useful knowledge and make appropriate decisions from big data , machine learning
    techniques have been regarded as a powerful solution .
  - In this paper we introduce Dex , the first continual reinforcement learning toolkit
    for training and evaluating continual learning methods . We present and demonstrate
    a novel continual learning method we call incremental learning to solve complex
    environments . In incremental learning , environments are framed as a task to
    be learned by an agent . This task can be split into a series of subtasks that
    are solved simultaneously . Similar to how natural language processing and object
    detection are subtasks of neural image caption generation [ 23 ] , reinforcement
    learning environments also have subtasks relevant to a given environment . These
    subtasks often include player detection , player control , obstacle detection
    , enemy detection , and player-object interaction , to name a few . These subtasks
    are common to many environments , but they are often sufficiently different in
    function and representation that reinforcement learning algorithms fail to generalize
    them across environments , such as in Atari . These critical subtasks are what
    expert humans utilize to quickly learn in new environments that share subtasks
    with previously learned environments , and are a reason for humans superior data
    efficiency in learning complex tasks .
  - 'Briefly , actuarial fairness demands that each policyholder should pay only for
    their own risk , that is , mutualization should occur only between individuals
    with the same ''true '' risk . In contrast , solidarity calls for equal contribution
    to the pool . On one level of this text , we problematize actuarial fairness (
    by extension , calibration ) as a notion of fairness in the normative sense by
    taking inspiration from insurance . This perspective is aligned with recent proposals
    that stress the discrepancy of formal algorithmic fairness and `` substantive
    '''' fairness ( Green , ) , which some prefer to call justice ( Vredenburgh ,
    ) . Parallel to this runs a distinct textual level , where we emphasize two intricately
    interacting themes : responsibility and tensions between aggregate and individual
    . Both entail criticism of actuarial fairness , but we suggest that they additionally
    provide much broader , fruitful lessons for machine learning from insurance .
    At the highest level of abstraction , our goal is to establish a general conceptual
    bridge between insurance and machine learning . Traversing this bridge , machine
    learning scholars can obtain new perspectives on the social situatedness of a
    probabilistic , statistical technology -we attempt to offer a new ''cognitive
    toolkit '' for thinking about the social situatedness of machine learning .'
- source_sentence: Which modeling and architectural design choices are especially
    important, and why?
  sentences:
  - We take a closer look at some theoretical challenges of Machine Learning as a
    function approximation , gradient descent as the default optimization algorithm
    , limitations of fixed length and width networks and a different approach to RNNs
    from a mathematical perspective .
  - This paper introduces Dex , a reinforcement learning environment toolkit specialized
    for training and evaluation of continual learning methods as well as general reinforcement
    learning problems . We also present the novel continual learning method of incremental
    learning , where a challenging environment is solved using optimal weight initialization
    learned from first solving a similar easier environment . We show that incremental
    learning can produce vastly superior results than standard methods by providing
    a strong baseline method across ten Dex environments . We finally develop a saliency
    method for qualitative analysis of reinforcement learning , which shows the impact
    incremental learning has on network attention .
  - 'Meanwhile , deep learning provides useful tools for processing and analyzing
    these big machinery data . The main purpose of this paper is to review and summarize
    the emerging research work of deep learning on machine health monitoring . After
    the brief introduction of deep learning techniques , the applications of deep
    learning in machine health monitoring systems are reviewed mainly from the following
    aspects : Autoencoder ( AE ) and its variants , Restricted Boltzmann Machines
    and its variants including Deep Belief Network ( DBN ) and Deep Boltzmann Machines
    ( DBM ) , Convolutional Neural Networks ( CNN ) and Recurrent Neural Networks
    ( RNN ) . Finally , some new trends of DL-based machine health monitoring methods
    are discussed .'
- source_sentence: How are machine learning methods applied to concrete scientific
    or engineering tasks (e.g., physics, biology, control, optimization)?
  sentences:
  - More specifically , it discusses six key challenge areas for software testing
    of machine learning systems , examines current approaches to these challenges
    and highlights their limitations . The paper provides a research agenda with elaborated
    directions for making progress toward advancing the state-of-the-art on testing
    of machine learning . Index termstesting challenges , machine learning , machine
    learning testing , testing ML , testing AI
  - As another example , if there is an anomalous drop in purchase of a product in
    an online store , it is possible that the product is out of stock , which needs
    attention . The state-of-the-art technique Fig . 1 . A New Framework for Anomaly
    Detection for anomaly detection is machine learning [ 4 ] , [ 7 ] , [ 9 ] , [
    11 ] , [ 12 ] , [ 14 ] . Machine learning techniques learn distributions on continuous
    variables . Anomaly events can be captured as deviations from established patterns
    ( distributions ) . However , there are certain temporal behaviors and relations
    that can not be easily learned by machine learning techniques , but can be easily
    characterized by formal languages such as PSL . In this paper , we propose a new
    framework called TEmporal Filtering ( TEF ) for anomaly detection ( Fig . 1 )
    . The idea is to merge machine learning with PSL monitors . The machine learning
    module takes as input a number of continuous variables x 1 , x 2 , . . . . . .
    , x m , and outputs some discrete events y 1 , y 2 , . . . . . . , y n , which
    become the input of the PSL monitor .
  - These critical subtasks are what expert humans utilize to quickly learn in new
    environments that share subtasks with previously learned environments , and are
    a reason for humans superior data efficiency in learning complex tasks . In the
    case of deliberately similar environments , we can construct the subtasks such
    that they are similar in function and representation that an agent trained on
    the first environment can accelerate learning on the second environment due to
    its preconstructed subtask representations , thus partially avoiding the more
    complex environment 's increased simulation cost and inherent learning difficulty
    .
- source_sentence: How do real-world constraints (e.g., deployment, cost, latency,
    privacy, safety) shape machine learning systems used for decision-making?
  sentences:
  - This is due to many different machine learning frameworks , computer architectures
    , and machine learning models . Historically , for modelling and simulation on
    HPC systems such problems have been addressed through benchmarking computer applications
    , algorithms , and architectures . Extending such a benchmarking approach and
    identifying metrics for the application of machine learning methods to scientific
    datasets is a new challenge for both scientists and computer scientists . In this
    paper , we describe our approach to the development of scientific machine learning
    benchmarks and review other approaches to benchmarking scientific machine learning
    .
  - Since 2006 , deep learning ( DL ) has become a rapidly growing research direction
    , redefining state-of-the-art performances in a wide range of areas such as object
    recognition , image segmentation , speech recognition and machine translation
    . In modern manufacturing systems , data-driven machine health monitoring is gaining
    in popularity due to the widespread deployment of low-cost sensors and their connection
    to the Internet . Meanwhile , deep learning provides useful tools for processing
    and analyzing these big machinery data . The main purpose of this paper is to
    review and summarize the emerging research work of deep learning on machine health
    monitoring .
  - This wider view on the entire machine learning field is largely ignored in the
    literature by keeping a strong focus entirely on models [ 2 ] . Our core contribution
    in this study is that we provide a clear view of the active research in machine
    learning by relying solely on a quantitative methodology without interviewing
    experts . This attempt aims at reducing bias and looking where the research community
    puts its focus on . The results of this study allow researchers to put their research
    into the global context of machine learning . This provides researchers with the
    opportunity to both conduct research in popular topics and identify topics that
    have not received sufficient attention in recent research . The rest of this paper
    is organized as follows . Section 2 describes the data sources and quantitative
    methodology . Section 3 presents and discusses the top 10 topics identified .
    Section 4 summarizes this work .
- source_sentence: What limitations are commonly identified, and what future directions
    or open problems follow from them?
  sentences:
  - Additionally , as environments become more complex , they will become more expensive
    to simulate . This poses a significant problem , since many Atari games already
    require upwards of 100 million steps using state-of-the-art algorithms , representing
    days of training on a single machine . Thus , it appears likely that complex environments
    will become too costly to learn from randomly initialized weights , due both to
    the increased simulation cost as well as the inherent difficulty of the task .
    Therefore , some form of prior information must be given to the agent . This can
    be seen with AlphaGo [ 18 ] , where the agent never learned to play the game without
    first using supervised learning on human games . While supervised learning certainly
    has been shown to aid reinforcement learning , it is very costly to obtain sufficient
    samples and requires the environment to be a task humans can play with reasonable
    skill , and is therefore impractical for a wide variety of important reinforcement
    learning problems . In this paper we introduce Dex , the first continual reinforcement
    learning toolkit for training and evaluating continual learning methods . We present
    and demonstrate a novel continual learning method we call incremental learning
    to solve complex environments . In incremental learning , environments are framed
    as a task to be learned by an agent .
  - As a consequence , the functional behavior expected from data-driven components
    can only be specified in part on their intended domain , and we can not assure
    that they will behave as expected in all cases . Moreover , their processing structure
    is usually difficult to trace and validate by humans because this structure rarely
    follows human intuition but is generated to provide the algorithmically generalized
    input-output relationship in an effective manner . Prominent representatives of
    models used by data-driven components are artificial neural networks and support
    vector machines ( Russell & Norvig , 2016 ) . Since data-driven models are an
    important source of uncertainty in embedded systems that collaborate in an open
    context , the uncertainty they introduce has to be appropriately understood and
    managed during design time and runtime . Previous work ( Kläs & Vollmer , 2018
    ) proposes separating the sources of uncertainty in data-driven components into
    three major classes , distinguishing between uncertainty caused by limitations
    in terms of model fit , data quality , and scope compliance . Whereas model fit
    focuses on the inherent uncertainty in data-driven models , data quality covers
    the additional uncertainty caused by their application to input data obtained
    in suboptimal conditions and scope compliance covers situations where the model
    is likely applied outside the scope for which it was trained and validated .
  - 'This means that for constant test inputs and preconditions , an ML-trained software
    component can produce different outputs in consecutive runs . Researchers have
    tried using testing techniques from traditional software development ( Hutchison
    et al . 2018 ) , to deal with some of these challenges . However , it has been
    observed that traditional testing approaches in general fail to adequately address
    fundamental challenges of testing ML ( Helle and Schamai 2016 ) , and that these
    traditional approaches require adaptation to the new context of ML . The better
    we understand current research challenges of testing ML , the more successful
    we can be in developing novel techniques that effectively address these challenges
    and advance this scientific field . In this paper , we : i ) identify and discuss
    the most challenging areas in software testing for ML , ii ) synthesize the most
    promising approaches to these challenges , iii ) spotlight their limitations ,
    and iv ) make recommendations of further research efforts on software testing
    of ML . We note that the aim of the paper is not to exhaustively list all published
    work , but distill the most representative work .'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'What limitations are commonly identified, and what future directions or open problems follow from them?',
    'This means that for constant test inputs and preconditions , an ML-trained software component can produce different outputs in consecutive runs . Researchers have tried using testing techniques from traditional software development ( Hutchison et al . 2018 ) , to deal with some of these challenges . However , it has been observed that traditional testing approaches in general fail to adequately address fundamental challenges of testing ML ( Helle and Schamai 2016 ) , and that these traditional approaches require adaptation to the new context of ML . The better we understand current research challenges of testing ML , the more successful we can be in developing novel techniques that effectively address these challenges and advance this scientific field . In this paper , we : i ) identify and discuss the most challenging areas in software testing for ML , ii ) synthesize the most promising approaches to these challenges , iii ) spotlight their limitations , and iv ) make recommendations of further research efforts on software testing of ML . We note that the aim of the paper is not to exhaustively list all published work , but distill the most representative work .',
    'As a consequence , the functional behavior expected from data-driven components can only be specified in part on their intended domain , and we can not assure that they will behave as expected in all cases . Moreover , their processing structure is usually difficult to trace and validate by humans because this structure rarely follows human intuition but is generated to provide the algorithmically generalized input-output relationship in an effective manner . Prominent representatives of models used by data-driven components are artificial neural networks and support vector machines ( Russell & Norvig , 2016 ) . Since data-driven models are an important source of uncertainty in embedded systems that collaborate in an open context , the uncertainty they introduce has to be appropriately understood and managed during design time and runtime . Previous work ( Kläs & Vollmer , 2018 ) proposes separating the sources of uncertainty in data-driven components into three major classes , distinguishing between uncertainty caused by limitations in terms of model fit , data quality , and scope compliance . Whereas model fit focuses on the inherent uncertainty in data-driven models , data quality covers the additional uncertainty caused by their application to input data obtained in suboptimal conditions and scope compliance covers situations where the model is likely applied outside the scope for which it was trained and validated .',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.2044, 0.3197],
#         [0.2044, 1.0000, 0.2727],
#         [0.3197, 0.2727, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 96 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 96 samples:
  |         | sentence_0                                                                         | sentence_1                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                               |
  | details | <ul><li>min: 15 tokens</li><li>mean: 23.29 tokens</li><li>max: 35 tokens</li></ul> | <ul><li>min: 46 tokens</li><li>mean: 199.49 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                             | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>What evaluation protocols and benchmark setups are commonly used to assess model performance?</code>                                             | <code>Among all common inquiries , perhaps the most basic is : can ML work on a specific problem ? Or , in other words : given the characteristics of a target data set , can the effectiveness of a ML approach be predicted ? Interestingly , this latter question can be further rephrased as : what are the characteristics of a data set that are well correlated with the possibility , or the impossibility , of obtaining ML models able to effectively extrapolate to unknown instances of the problem ? It is well known that ML algorithms are affected by the curse of dimensionality [ 11 ] , but ML practitioners also know that it could be possible to obtain reliable models even for high-dimensional data sets , and with a relatively small number of samples [ 12 ] . The common approach among practitioners in the field , when dealing with a new data set , seems to be : try as many different ML algorithms as possible in a cross-validation , and evaluate the outcomes ; then focus on the techniques that provi...</code> |
  | <code>What is the central idea or key contribution of a machine learning approach?</code>                                                              | <code>As another example , if there is an anomalous drop in purchase of a product in an online store , it is possible that the product is out of stock , which needs attention . The state-of-the-art technique Fig . 1 . A New Framework for Anomaly Detection for anomaly detection is machine learning [ 4 ] , [ 7 ] , [ 9 ] , [ 11 ] , [ 12 ] , [ 14 ] . Machine learning techniques learn distributions on continuous variables . Anomaly events can be captured as deviations from established patterns ( distributions ) . However , there are certain temporal behaviors and relations that can not be easily learned by machine learning techniques , but can be easily characterized by formal languages such as PSL . In this paper , we propose a new framework called TEmporal Filtering ( TEF ) for anomaly detection ( Fig . 1 ) . The idea is to merge machine learning with PSL monitors . The machine learning module takes as input a number of continuous variables x 1 , x 2 , . . . . . . , x m , and outputs some discr...</code> |
  | <code>How do real-world constraints (e.g., deployment, cost, latency, privacy, safety) shape machine learning systems used for decision-making?</code> | <code>Since 2006 , deep learning ( DL ) has become a rapidly growing research direction , redefining state-of-the-art performances in a wide range of areas such as object recognition , image segmentation , speech recognition and machine translation . In modern manufacturing systems , data-driven machine health monitoring is gaining in popularity due to the widespread deployment of low-cost sensors and their connection to the Internet . Meanwhile , deep learning provides useful tools for processing and analyzing these big machinery data . The main purpose of this paper is to review and summarize the emerging research work of deep learning on machine health monitoring .</code>                                                                                                                                                                                                                                                                                                                                              |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 1
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.13.1
- Sentence Transformers: 5.2.0
- Transformers: 4.57.6
- PyTorch: 2.9.1+cpu
- Accelerate: 1.12.0
- Datasets: 4.5.0
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->