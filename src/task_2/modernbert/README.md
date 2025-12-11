First of all, you should have the MADON dataset already uploaded on HuggingFace, as mentioned before. Then, you will find some important scripts in here:

- ``preprocess.py`` : loads the MADON dataset from HuggingFace, preprocesses it and uploads it again to HuggingFace. This is, so the preprocessing is done just once for all following finetuning tasks, which just load the preprocessed data from HuggingFace.
- ``finetune_large.py`` : finetunes ModernBERT-large with the dataset from HuggingFace and uploads the finetuned model again on HuggingFace. Also logs data to Comet.
- ``inference.py`` : Loads the finetuned model and the preprocessed data both from HuggingFace to make predictions on the test-dataset. Although some evaluation is done in these scripts, the general evaluation script is in the folder above, as mentioned earlier. Therefore, CSV files are created with this script, which can be used for the main evaluation script.

We are using conda for managing environments.

After installing conda, you can directly recreate the environment with the yaml file:

``conda env create -f environment.yml``

If you already have an environement called ``madon_env``, then you can change the first line of the ``environment.yml`` file to a new name.

Then activate it:

``conda activate madon_env``

Before preprocessing the data, you must set up 3 variables in the ``preprocess.py`` file:

- ``dataset_src`` : the HuggingFace link to the MADON dataset
- ``config`` : config for the dataset_src. Set it to ``None`` if you don't have a config name.
- ``dataset_dest`` : where to save the preprocessed dataset on HuggingFace

You can run preprocessing with:

``python preprocess.py``

After preprocessing, you should be ready to finetune the model. Since we are using comet to save training logs, you should either change the code for your logging platform, or input your comet credentials in the code. So you must change the following in the ``finetune_large.py`` file:

- Comet login credentials:
	- ``api_key``
	- ``project_name``
	- ``workspace``
- ``checkpoint_dest`` : the HuggingFace reference, where to save the finetuned model after finetuning.
- ``dataset_id`` : the link to HuggingFace where you saved the preprocessed dataset.

Now you're ready for finetuning. You can configure your GPU setup for finetuning with accelerate:

``accelerate config``

and finally start the finetuning:

``accelerate launch finetune_large.py``

After finetuning is finished, you can now create the CSV files for evaluation. So now, you must run the ``inference.py`` script, but again, you have to set up some variables in that file:

- ``model_id`` : link to the finetuned model on HuggingFace
- ``revision_id`` : the revision ID that you want to use. If you want to just use the latest one, you can set it to ``"main"``
- ``dataset_id`` : the HuggingFace link to the preprocessed dataset.

You can now run the script with 

``python inference.py``

and will get CSV files with the inference. You can now use these files for evaluation.