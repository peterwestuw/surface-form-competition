# Surface Form Competition

This is the official repo of the paper ["Surface Form Competition: Why the Highest Probability Answer Isn't Always Right"](https://peterwestuw.github.io/surface-form-competition-project/) We provide scripts for downloading/processing datasets and for reproducing our results on GPT-2 and GPT-3. We do not guarantee exact reproducibility, as library versions and GPUs may cause small differences, but these should be extremely minor.

## Dependencies
We use python3 and pytorch 1.7.0, but we do not use cutting-edge features from either and expect to be largely forward and backward compatible. That is not a guarantee or promise.

You can use `pip install -r requirements.txt` to install the required libraries.

## OpenAI Beta
To use GPT-3 you must use OpenAI Beta, which is limited access. You can apply for access [here](https://beta.openai.com/). Once you have access you will need to point the `score.py` to your API key with the `--key` argument or put your key in `api.key` which is the default path. 

## Downloading Datasets

`DATA_README.md` has thorough instructions for downloading and processing datasets. We provide automatic downloaders and processers for datasets where possible in `data_downloaders/` but see `DATA_README` for full instructions.

## Running Scorers
Once you have a dataset downloaded, running all the zero-shot scoring strategies at once is as simple as:

```
python score.py <dataset abbrevation> --model <model>
```

where `<dataset-abbreviation>` is the abbreviation for a given dataset used for table rows in the paper. If there is any confusion, simply look in `score.py` to see how dataset selection works. `<model>` is the name of either a GPT-2 or GPT-3 model e.g. `xl`, `davinci`, etc. To speed things up you can use a larger `--batch` if you have enough GPU memory.
