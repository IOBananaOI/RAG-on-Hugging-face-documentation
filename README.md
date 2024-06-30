# RAG-on-Hugging-face-documentation

### Implementation of RAG-system using 🤗 Hugging face documentation dataset and LangChain library.

This repository contains fully completed RAG-system for answering the questions from Hugging face documentation.

### Example of use:

First argument is the question itself and the second is the folder, where knowledge vector database is located.

``` sh
python reader.py "How to make a pipeline?" "knowledge_vector_base"
```

```
To make a pipeline, follow these steps:

1. Decide on the raw inputs and outputs of the pipeline. These could be strings, dictionaries, or any other format that is easy to work with.

2. Inherit the `Pipeline` class from the `transformers` module and implement the following methods:
   - `preprocess`: Preprocesses the input data into a format that the model expects.
   - `_forward`: Runs the actual inference using the model and scheduler.
   - `postprocess`: Postprocesses the output of the model to make it more human-readable.
   - `_sanitize_parameters`: Sanitizes the parameters passed to the pipeline to ensure they are compatible with the underlying model and scheduler.

3. Load the necessary models and schedulers for your pipeline type. These can be loaded automatically by passing a checkpoint to the pipeline constructor.

4. Use the pipeline to perform inference on new data. Simply call the `__call__` method of the pipeline object with the input data.

Here's an example implementation for a simple text classification pipeline:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines import Pipeline

model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base")
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

class TextClassificationPipeline(Pipeline):
    def __init__(self, model=None, tokenizer=None, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, **kwargs)

    def preprocess(self, inputs):
        return self.tokenizer(inputs, padding="max_length", truncation=True, return_tensors="pt").input_ids

    def _forward(self, inputs):
        return self.model(inputs)[0]

    def postprocess(self, logits):
        return self.model.config.id2label[torch.argmax(logits).item()]

    def _sanitize_parameters(self, params):
        # Check that batch size is not too large
        if "batch_size" in params and params["batch_size"] > 128:
            params["batch_size"] = 128
        return params

text_classifier = TextClassificationPipeline()
predictions = text_classifier("This is a great product!")
print(predictions)

This pipeline uses the DistilRoberta model for text classification, and returns the predicted label as a string. You can replace the model and tokenizer with your own choices, and modify the preprocessing, forward pass, and postprocessing functions to suit your specific use case.
```


To create knowledge vector database use this command.

``` sh
python knowledge_base.py --folder_path=knowledge_vector_base
```

## Knowledge database vector representation

Star represents the vector embedding of user query "How to make a pipeline?"

<img src='https://github.com/IOBananaOI/RAG-on-Hugging-face-documentation/blob/main/images/knowledge_database_representation.png?raw=true'>
