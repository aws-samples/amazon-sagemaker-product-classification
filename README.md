## Product classification - creating a classifier for categories based on other metadata

Some companies want to create their own taxonomy where they have a large dataset from various sources but they don't have a consistent category or label for that data.
Their dataset will typically have other columns/features which could be used to inform that decision.

In this example here, I have downloaded a large public dataset of books which has many columns that I disregarded and I just focused on the title and description fields as I think they will have the most influence on the book category.

At a high level, what this notebook is doing is using a word embedder to vectorize the sentences for description and title. By doing this, we are effectively capturing the semantics of the sentence and as a result able to infer meaning.
With the vector and the labelled dataset, we can then train a classifier to predict the category of the book by just using the vector as an input.

For this particular example, we had some class imbalance which typically is bad for the model training as the model will favour the majority class and is likely to predict the majority class every time. XGBoost has a version of the algorithm where you can supply weights for the classes which significantly improves the performance of the model.
https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier  
This means that we cannot use the built-in XGBoost algorithm used in SageMaker and therefore shows an example of how you can bring your own script, and still use the XGBoost framework so you avoid creating your own docker image and can still do your training and deployment of your model in the cloud.

There are some links here on how to implement both the gensim implementation of FastText and then the link below on how to implement the native method.
https://radimrehurek.com/gensim/models/fasttext.html
https://fasttext.cc/docs/en/support.html

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
