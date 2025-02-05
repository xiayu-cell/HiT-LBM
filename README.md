## Code for Hierarchical Tree Search-based User Lifelong Behavior Modeling on Large Language Model

## Setup

1. Download dataset
   
   Take Amazon-Books for example, download the dataset to folder `data/amz/raw_data/`
2. Preprocessing: in folder `preprocess`
   1. run `python preprocess_amz.py`.
   2. run `generate_data_and_prompt.py` to generate data for CTR task, as well as prompt for LLM.
   
3. User Interest generation
   1. We provide the pre-generated user interest chunks and their corresponding BGE embeddings. The user interests are stored in JSON format, including the interests of each user for each chunk and the evolution of these interests over time with Hierarchical Tree Search.
   You can be access from [Here](https://drive.google.com/drive/folders/1OdL6JPq_UZUSCO3skAIX3NOxF81goB3F?usp=sharing)

   2. You can also use your own LLM to generate user interests.
      1. `cd preference_generation`
      2. `python batch_infer.py`
      3. `python batch_infer_item.py`

4. Hierarchical Tree Search
   1. Process Rating Model
      1. You can run the following file to achieve the expansion of interest nodes:
         `python batch_infer_with_Best_N.py`
      2. Utilize LLM to score for continuity (SRM) and validity (PRM):
         1. `python SRM/batch_infer.py`
         2. `python PRM/batch_infer_book.py`
      3. Train SRM and PRM:
         1. `python SRM/prm.py`
         2. `python PRM/prm_amz.py`
   
   2. Inference
      1. `python PRM/batch_infer_with_Best_N_amz.py`

5. Knowledge encoding: in folder `knowledge_encoding`
   1. Run `python lm_encoding_with_item_bge.py`

6. RS: in folder `RS`
   1. `bash RS/run_amz.sh` for ctr task

Our implementation code is based on : 
[KAR](https://github.com/YunjiaXi/Open-World-Knowledge-Augmented-Recommendation/tree/main)
