import sys
sys.path.append('../../')
import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
import wandb
wandb.init(project="trl_wandb")

df = pd.read_csv('../data/imdb-dataset.csv')

df.head()

df['label'] = (df['sentiment']=='positive').astype(int)

df.head()

df.rename({'review': 'text'}, axis=1, inplace=True)
df.drop('sentiment', axis=1, inplace=True)

df.head()

df_train, df_valid = train_test_split(df, test_size=0.2)

args = {
    'fp16':False,
    'wandb_project': 'bert-imdb',
    'num_train_epochs': 3,
    'overwrite_output_dir':True,
    'learning_rate': 1e-5,
}

model = ClassificationModel('bert', 'bert-large-cased', use_cuda=True,args=args) 
model.train_model(df_train, output_dir='bert-imdb')
result, model_outputs, wrong_predictions = model.eval_model(df_valid)

(result['tp']+result['tn'])/(result['tp']+result['tn']+result['fp']+result['fn'])

model.predict(['The movie was really good'])
