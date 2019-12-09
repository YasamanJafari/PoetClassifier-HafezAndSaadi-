
# CA3 - Naive Bayes Classification
## Yasaman Jafari
### 810195376

In this project, Naive Bayes is used to classify poems of two persian poets, "Hafez" and "Saadi".


```python
import pandas as pd
import re
import math
```

First, we read the data from .csv file.


```python
data = pd.read_csv("./Data/train_test.csv", encoding="utf-8")
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>چون می‌رود این کشتی سرگشته که آخر</td>
      <td>hafez</td>
    </tr>
    <tr>
      <th>1</th>
      <td>که همین بود حد امکانش</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ارادتی بنما تا سعادتی ببری</td>
      <td>hafez</td>
    </tr>
    <tr>
      <th>3</th>
      <td>خدا را زین معما پرده بردار</td>
      <td>hafez</td>
    </tr>
    <tr>
      <th>4</th>
      <td>گویی که در برابر چشمم مصوری</td>
      <td>saadi</td>
    </tr>
  </tbody>
</table>
</div>



As you can see above, we are provided with about 20000 samples and each one is labeled with its poet.


```python
print(data["text"][0])
```

    چون می‌رود این کشتی سرگشته که آخر


First, we need to separate our data into two parts of train set and test set. Eighty percent of data is considered as train set.
We need to shuffle the data and then choose 80% for train and leave the rest for test.

We choose a random subset for train and test so that it is not sorted in any specific order and it can represent the entire poems collection better.


```python
train = data.sample(frac = 0.8, random_state = 42)
test = data.drop(train.index)
```


```python
print("Train Percentage: ", len(train) / (len(test) + len(train)))
print("Test Percentage: ", len(test) / (len(test) + len(train)))
```

    Train Percentage:  0.7999904255828426
    Test Percentage:  0.20000957441715736



```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3128</th>
      <td>سه ماه می خور و نه ماه پارسا می‌باش</td>
      <td>hafez</td>
    </tr>
    <tr>
      <th>8157</th>
      <td>زاهد بنگر نشسته دلتنگ</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>6682</th>
      <td>ولیکن تا به چوگان می‌زنندش</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>11526</th>
      <td>تا فخر دین عبدالصمد باشد که غمخواری کند</td>
      <td>hafez</td>
    </tr>
    <tr>
      <th>7477</th>
      <td>تیغ جفا گر زنی ضرب تو آسایشست</td>
      <td>saadi</td>
    </tr>
  </tbody>
</table>
</div>



Now we need to find all the words in the poems. For this I created a new column which keeps the list of words used in each poem.


```python
train['index'] = train.index
train['words'] = train.text.str.split().to_frame()
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
      <th>index</th>
      <th>words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3128</th>
      <td>سه ماه می خور و نه ماه پارسا می‌باش</td>
      <td>hafez</td>
      <td>3128</td>
      <td>[سه, ماه, می, خور, و, نه, ماه, پارسا, می‌باش]</td>
    </tr>
    <tr>
      <th>8157</th>
      <td>زاهد بنگر نشسته دلتنگ</td>
      <td>saadi</td>
      <td>8157</td>
      <td>[زاهد, بنگر, نشسته, دلتنگ]</td>
    </tr>
    <tr>
      <th>6682</th>
      <td>ولیکن تا به چوگان می‌زنندش</td>
      <td>saadi</td>
      <td>6682</td>
      <td>[ولیکن, تا, به, چوگان, می‌زنندش]</td>
    </tr>
    <tr>
      <th>11526</th>
      <td>تا فخر دین عبدالصمد باشد که غمخواری کند</td>
      <td>hafez</td>
      <td>11526</td>
      <td>[تا, فخر, دین, عبدالصمد, باشد, که, غمخواری, کند]</td>
    </tr>
    <tr>
      <th>7477</th>
      <td>تیغ جفا گر زنی ضرب تو آسایشست</td>
      <td>saadi</td>
      <td>7477</td>
      <td>[تیغ, جفا, گر, زنی, ضرب, تو, آسایشست]</td>
    </tr>
  </tbody>
</table>
</div>



In order to find all the words used in the set, I combined all the lists and then converted it to a set in order to remove duplicates.

Some words are commonly used in persian and are not useful in classification.


```python
stop_words = ['دل', 'گر', 'ما', 'هر', 'با', 'ای', 'سر', 'تا', 'چو', 'نه']
```

Now, I'll separate train data of Hafez and Saadi to do some process on them.


```python
hafez_train = train[train['label'] == "hafez"].drop(['text', 'label', 'index'], axis=1)
saadi_train = train[train['label'] == "saadi"].drop(['text', 'label', 'index'], axis=1)

print("Hafez Count: ", len(hafez_train))
print("Saadi Count: ", len(saadi_train))
```

    Hafez Count:  6753
    Saadi Count:  9958


I'll consider each word as a feature. For this purpose, I need to find all distinct words in the train set.


```python
words = []
for poem in train['words']:
    words += poem
words = list(set(words))
print("distinct_words_count: ", len(words))
```

    distinct_words_count:  12645


find_prob() function is used to we need to find the conditional probability of each word given the poet. For each word we need to find the number of occurances of it in a poet's poems and divide it by the total number of words used by that poet.


```python
def find_prob(hafez_train, saadi_train):
    hafez_words = []
    for poem in hafez_train['words']:
        hafez_words += poem
    print("Hafez_count: ", len(hafez_words))
    
    saadi_words = []
    for poem in saadi_train['words']:
        saadi_words += poem
    print("Saadi_count: ", len(saadi_words))
    
    train_all_word_count = pd.DataFrame(columns=['hafez_count', 'saadi_count'])
    for word in words:
        train_all_word_count = train_all_word_count.append({'word': word, 'hafez_count': hafez_words.count(word), 'saadi_count': saadi_words.count(word)}, ignore_index=True)
        
    train_all_word_count = train_all_word_count.set_index('word')
    train_all_word_count['hafez_prob'] = train_all_word_count['hafez_count'] / train_all_word_count['hafez_count'].sum()
    train_all_word_count['saadi_prob'] = train_all_word_count['saadi_count'] / train_all_word_count['saadi_count'].sum()
    
    return train_all_word_count, hafez_words, saadi_words
```


```python
train_all_word_count, hafez_words, saadi_words = find_prob(hafez_train, saadi_train)
```

    Hafez_count:  50650
    Saadi_count:  70650



```python
train_all_word_count.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hafez_count</th>
      <th>saadi_count</th>
      <th>hafez_prob</th>
      <th>saadi_prob</th>
    </tr>
    <tr>
      <th>word</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>دلبند</th>
      <td>1</td>
      <td>8</td>
      <td>1.97433e-05</td>
      <td>0.000113234</td>
    </tr>
    <tr>
      <th>بدبین</th>
      <td>1</td>
      <td>0</td>
      <td>1.97433e-05</td>
      <td>0</td>
    </tr>
    <tr>
      <th>احوال</th>
      <td>7</td>
      <td>4</td>
      <td>0.000138203</td>
      <td>5.66171e-05</td>
    </tr>
    <tr>
      <th>خودپسند</th>
      <td>1</td>
      <td>0</td>
      <td>1.97433e-05</td>
      <td>0</td>
    </tr>
    <tr>
      <th>شهد</th>
      <td>3</td>
      <td>7</td>
      <td>5.923e-05</td>
      <td>9.908e-05</td>
    </tr>
  </tbody>
</table>
</div>



The prior probabilities are calculated below. The prior probability of each poets in the total number of that poet's poems divided by all poems by both poets.


```python
hafez_prob = len(hafez_train) / (len(hafez_train) + len(saadi_train))
saadi_prob = len(saadi_train) / (len(hafez_train) + len(saadi_train))

print("Hafez Probability: ", hafez_prob)
print("Saadi Probability: ", saadi_prob)
```

    Hafez Probability:  0.4041050804859075
    Saadi Probability:  0.5958949195140926


At this step, we will eliminate the words which are only used once by Hafez or Saadi as they cannot be distinctive.


```python
# train_all_word_count['all_count'] = train_all_word_count['hafez_count'] + train_all_word_count['saadi_count']
# one_occurance = train_all_word_count[train_all_word_count['all_count'] == 1]
# once_used = list(one_occurance.index)
# words = list(set(words) - set(once_used))

# train_all_word_count = find_prob()
```

# Operate On Test Data

Now we have built our model and need to predict the poet of the test data.


```python
test['index'] = test.index
test['words'] = test.text.str.split().to_frame()
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
      <th>index</th>
      <th>words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>رفتی و همچنان به خیال من اندری</td>
      <td>saadi</td>
      <td>9</td>
      <td>[رفتی, و, همچنان, به, خیال, من, اندری]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>آنجا که تویی رفتن ما سود ندارد</td>
      <td>saadi</td>
      <td>11</td>
      <td>[آنجا, که, تویی, رفتن, ما, سود, ندارد]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>اندرونم با تو می‌آید ولیک</td>
      <td>saadi</td>
      <td>13</td>
      <td>[اندرونم, با, تو, می‌آید, ولیک]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>که خوش آهنگ و فرح بخش هوایی دارد</td>
      <td>hafez</td>
      <td>16</td>
      <td>[که, خوش, آهنگ, و, فرح, بخش, هوایی, دارد]</td>
    </tr>
    <tr>
      <th>24</th>
      <td>ناودان چشم رنجوران عشق</td>
      <td>saadi</td>
      <td>24</td>
      <td>[ناودان, چشم, رنجوران, عشق]</td>
    </tr>
  </tbody>
</table>
</div>



In Naive Bayes, we have the strong assumption of independence between each two features.

So, the probability of each poet given the words is proportional to the multiplication of all the probabilities of each word given the poet multiplied by the prior probability which is the probability of each poet in general.

After calculating this probability for Hafez and Saadi, we compare them and decide based on the result.


```python
def predict(df):
    for index, row in df.iterrows(): 
        curr_hafez_prob = len(hafez_train) / (len(hafez_train) + len(saadi_train))
        curr_saadi_prob = len(saadi_train) / (len(hafez_train) + len(saadi_train))
        for word in set(row["words"]):
            if word in words:
                curr_hafez_prob *= train_all_word_count.at[word, 'hafez_prob']
                curr_saadi_prob *= train_all_word_count.at[word, 'saadi_prob']
        df.at[index, 'hafez_prob'] = curr_hafez_prob
        df.at[index, 'saadi_prob'] = curr_saadi_prob

    df['prediction_is_hafez'] = df['hafez_prob'] >= df['saadi_prob']

    prediction_poet = {True: 'hafez', False: 'saadi'}
    df['prediction'] = df['prediction_is_hafez'].map(prediction_poet)
```


```python
predict(test)
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
      <th>index</th>
      <th>words</th>
      <th>hafez_prob</th>
      <th>saadi_prob</th>
      <th>prediction_is_hafez</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>رفتی و همچنان به خیال من اندری</td>
      <td>saadi</td>
      <td>9</td>
      <td>[رفتی, و, همچنان, به, خیال, من, اندری]</td>
      <td>0.000000e+00</td>
      <td>6.067397e-21</td>
      <td>False</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>11</th>
      <td>آنجا که تویی رفتن ما سود ندارد</td>
      <td>saadi</td>
      <td>11</td>
      <td>[آنجا, که, تویی, رفتن, ما, سود, ندارد]</td>
      <td>2.686915e-23</td>
      <td>5.948826e-22</td>
      <td>False</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>13</th>
      <td>اندرونم با تو می‌آید ولیک</td>
      <td>saadi</td>
      <td>13</td>
      <td>[اندرونم, با, تو, می‌آید, ولیک]</td>
      <td>0.000000e+00</td>
      <td>1.970144e-16</td>
      <td>False</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>16</th>
      <td>که خوش آهنگ و فرح بخش هوایی دارد</td>
      <td>hafez</td>
      <td>16</td>
      <td>[که, خوش, آهنگ, و, فرح, بخش, هوایی, دارد]</td>
      <td>7.862324e-25</td>
      <td>0.000000e+00</td>
      <td>True</td>
      <td>hafez</td>
    </tr>
    <tr>
      <th>24</th>
      <td>ناودان چشم رنجوران عشق</td>
      <td>saadi</td>
      <td>24</td>
      <td>[ناودان, چشم, رنجوران, عشق]</td>
      <td>4.217595e-06</td>
      <td>8.562444e-06</td>
      <td>False</td>
      <td>saadi</td>
    </tr>
  </tbody>
</table>
</div>



In order to evaluate how good our model is, we use:
* Recall
* Precision
* Accuracy


```python
def evaluate(df):
    df['correct'] = (df['label'] == df['prediction'])
    correct_count = (df['correct']).sum()
    correct_hafez = (df[['correct', 'prediction_is_hafez']].all(axis='columns')).sum()
    all_hafez = (df['label'] == 'hafez').sum()
    all_hafez_detected = (df['prediction'] == 'hafez').sum()
    accuracy = correct_count / len(test)
    precision = correct_hafez / all_hafez_detected
    recall = correct_hafez / all_hafez
    print("Recall: ", recall)
    print("Precision: ", precision)
    print("Accuracy: ", accuracy)
```


```python
evaluate(test)
```

    Recall:  0.7225225225225225
    Precision:  0.7229567307692307
    Accuracy:  0.7790808999521303


## Laplace Smoothing

If a word is used only in one poet's work, the probability of it given the other poet, will be zero and as we multiply the probabilities, the result will be zero not considering any other features.

In order to fix this, we add a fixed alpha to the count of all words in its poet's collection and add count * alpha  to the denominator.


```python
alpha = 0.5
train_all_word_count['hafez_prob'] = (train_all_word_count['hafez_count'] + alpha) / (train_all_word_count['hafez_count'].sum() + (len(set(hafez_words + saadi_words))* alpha))
train_all_word_count['saadi_prob'] = (train_all_word_count['saadi_count'] + alpha) / (train_all_word_count['saadi_count'].sum() + (len(set(saadi_words + hafez_words))* alpha))
```


```python
print(train_all_word_count['hafez_prob'].sum())
print(train_all_word_count['saadi_prob'].sum())
```

    1.000000000000271
    1.000000000000061



```python
predict(test)
```


```python
evaluate(test)
```

    Recall:  0.7375375375375376
    Precision:  0.7767235926628716
    Accuracy:  0.8109143130684539


## Evaluate


```python
data['index'] = data.index
data['words'] = data.text.str.split().to_frame()

hafez_data = data[data['label'] == "hafez"].drop(['text', 'label', 'index'], axis=1)
saadi_data = data[data['label'] == "saadi"].drop(['text', 'label', 'index'], axis=1)
words = []
for poem in data['words']:
    words += poem
words = list(set(words))
print("distinct_words_count: ", len(words))

train_all_word_count, hafez_words, saadi_words = find_prob(hafez_data, saadi_data)

eval_data = pd.read_csv("./Data/evaluate.csv", encoding="utf-8")

alpha = 0.5
train_all_word_count['hafez_prob'] = (train_all_word_count['hafez_count'] + alpha) / (train_all_word_count['hafez_count'].sum() + (len(set(hafez_words + saadi_words)) * alpha))
train_all_word_count['saadi_prob'] = (train_all_word_count['saadi_count'] + alpha) / (train_all_word_count['saadi_count'].sum() + (len(set(hafez_words + saadi_words)) * alpha))

eval_data['index'] = eval_data.id
eval_data['words'] = eval_data.text.str.split().to_frame()

predict(eval_data)
```

    distinct_words_count:  14084
    Hafez_count:  63077
    Saadi_count:  88560



```python
eval_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>text</th>
      <th>index</th>
      <th>words</th>
      <th>hafez_prob</th>
      <th>saadi_prob</th>
      <th>prediction_is_hafez</th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>ور بی تو بامداد کنم روز محشر است</td>
      <td>1</td>
      <td>[ور, بی, تو, بامداد, کنم, روز, محشر, است]</td>
      <td>3.001295e-26</td>
      <td>7.121545e-24</td>
      <td>False</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>ساقی بیار جامی کز زهد توبه کردم</td>
      <td>2</td>
      <td>[ساقی, بیار, جامی, کز, زهد, توبه, کردم]</td>
      <td>2.353900e-23</td>
      <td>1.252923e-25</td>
      <td>True</td>
      <td>hafez</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>مرا هرآینه خاموش بودن اولی‌تر</td>
      <td>3</td>
      <td>[مرا, هرآینه, خاموش, بودن, اولی‌تر]</td>
      <td>3.377382e-22</td>
      <td>2.224892e-20</td>
      <td>False</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>تو ندانی که چرا در تو کسی خیره بماند</td>
      <td>4</td>
      <td>[تو, ندانی, که, چرا, در, تو, کسی, خیره, بماند]</td>
      <td>4.951443e-25</td>
      <td>6.114006e-23</td>
      <td>False</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>کاینان به دل ربودن مردم معینند</td>
      <td>5</td>
      <td>[کاینان, به, دل, ربودن, مردم, معینند]</td>
      <td>1.256054e-23</td>
      <td>5.174989e-22</td>
      <td>False</td>
      <td>saadi</td>
    </tr>
  </tbody>
</table>
</div>




```python
output = pd.DataFrame({
    "id": eval_data['index'],
    "label": eval_data['prediction'],
})
output.to_csv('output.csv', index=False)
```


```python
output.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>hafez</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>saadi</td>
    </tr>
  </tbody>
</table>
</div>



## Report Questions and Explanations

### Parameters

In my algorithm, each distinct word is considered a feature.

Bayesian probability consists of 4 parts:
* Prior
* Posterior
* Likelihood
* Evidence

$P(Poet|W_0, W_1, W_2, ...,  W_n) = \frac{P(Poet)P(W_0, W_1, W_2, ...,  W_n|Poet)}{P(W_0, W_1, W_2, ...,  W_n)}$

In the equation above, we have:
* Prior: P(Poet)
* Likelihood: P(W0, W1, W2, ..., Wn|Poet)
* Evidence: P(W0, W1, W2, ..., Wn)
* Posterior: P(Poet|W0, W1, W2, ..., Wn)

In other words, Prior is the probability of each poet in general, this means how probable it is for a poem to be for a certain poet in general not considering any other data. In order to calculate the prior, for each poet, we divide the number of that poet's poems by the number of all given poems.
 
Likelihood is the probability of each word of a poem given the poet. In Naive Bayes, each feature is independent of others, so this is the multiplication of the probabilities of each word given the poet. This means how probable it is for a certain poet to use that word. To calculate this, we multiply the probabilities of each word given the poet. The probaility of each word given the poet is the number of times that word is used in that poet's works divided by the total number of words in that poet's work.

Evidence is the probabiliy of all words that we have in a given poem. We do not need to calculate this as it is the same for both poets and does not change the result of comparison. If we wanted to calculate this, we could multiply the probabilites of all words. The probability of each word is the number of its occurance divided by the count of all words.

Posterior is the probaility of a poet given the words in a poem. We use bayesian rule stated below to calculate this.

$$ P(c|X) = \frac{P(c)\times\prod_{i=1}^{m} P(x_i|c)}{P(X)} $$

### Extra Questions

**1. What is the problem if we only use precision to evaluate our model?**

If we only use precision for model evaluation, we can get a 100 percent precision if we manage to correctly guess only one poem of the corresponding poet, and predict the other poet for all other given poems.

$$ Precision = \frac{True Positive}{True Positive + False Positive} $$

In other words if we are able to predict one of Hafez's poems correctly and assign it to Hafez and then assign all the other poems to Saadi, the precision for Hafez will be 100%.

**2. Why isn't accuracy enough for evaluating the model?**

If the majority of the data belongs to a specific class, accuracy will not be a good measure to evaluate our model. 
For instance if we want to predict if a person has cancer or not, the majority of people do not have cancer and we can simply predict that no one has cancer and we will get a high accuracy as we have detected almost every case correctly but the model is not good at all.

### Laplace 

Having a word which only exists in one of the poet's work in the training data, the probaility of that word given the other poet will be zero and as we consider the multiplication of these probabilities, the result will be zero ignoring all other probabilities, so the given poet will not be assigned to this poet. 

In order to fix this, I added a small alpha to the count of each word while calculating the corresponding probability. I also added distinct_count * alpha to denominator in order to have the sum of 1 for the new probabilities.

For instance the percentages before laplace is shown below in one case:

* Recall:  0.7213213213213213
* Precision:  0.7200239808153477
* Accuracy:  0.7771661081857348

After Laplace:

* Recall:  0.7645645645645646
* Precision:  0.755938242280285
* Accuracy:  0.8078027764480613

As you can see above, all the percentages have improved.


```python
data_1 = pd.read_csv("./output.csv", encoding="utf-8")
data_2 = pd.read_csv("/Users/yasaman/Desktop/yasaman.csv", encoding="utf-8")
```


```python
data_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>hafez</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>saadi</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>hafez</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>saadi</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>saadi</td>
    </tr>
  </tbody>
</table>
</div>




```python
(data_1 == data_2)['label'].
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>9</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>18</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>20</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>27</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>28</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1052</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1053</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1055</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1056</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1057</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1059</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1060</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1061</th>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1062</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1063</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1064</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1065</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1066</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1067</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1068</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1069</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1070</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1071</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1072</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1073</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1074</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1075</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1076</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1077</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1078</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1079</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1080</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1081</th>
      <td>True</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>1082 rows × 2 columns</p>
</div>




```python
len(data_1)
```




    1082




```python

```
