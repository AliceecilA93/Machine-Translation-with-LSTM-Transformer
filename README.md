# Machine-Translation-with-LSTM and Transformer
 
## 진행기간 
- 2022.09.02. ~ 2022.09.13

## 목적
- **논문을 기반으로 Seq2Seq 모델,Attention기법, Transformer를 사용함으로써 기계번역 성능 향상**  
          
## 코드 설명

   
코드     | 코드 링크   | 
:-------:|:-----------:|
Seq2Seq(LSTM)|[Seq2Seq(LSTM)](https://github.com/AliceecilA93/Machine-Translation-with-LSTM-and-Transformer/blob/main/Seq2Seq(LSTM).ipynb)|         
Seq2Seq with Attention | [Seq2Seq with Attention](https://github.com/AliceecilA93/Machine-Translation-with-LSTM-and-Transformer/blob/main/Seq2Seq%20with%20Attention.ipynb)|
Seq2Seq with Attention(Bi-LSTM)| [Seq2Seq with Attention(Bi-LSTM)](https://github.com/AliceecilA93/Machine-Translation-with-LSTM-and-Transformer/blob/main/Seq2Seq%20with%20Attention(Bi-LSTM).ipynb)| 
Transformer| [Transformer](https://github.com/AliceecilA93/Machine-Translation-with-LSTM-and-Transformer/blob/main/Transformer.ipynb) |
        

## 사용된 데이터  

- AI Hub [1_구어체(1)](https://drive.google.com/uc?id=1V6HsBoEczDoo4NDZ1I5iXSfRxFxCatis)


## 사용된 모델 

- LSTM
- Bi-LSTM
- Transformer


## 과정  

 1. 개발환경 : Python, Tensorflow, Colab
 
 2. 데이터 전처리
    - 형태소 분석기 비교( Komoran, Mecab, Hannanum, Okt, Kkma) 
    - preprocess 
```c
def kor_preprocess(sent):
  sent = re.sub(r"([?.!,¿])", r" \1", sent)
  sent = re.sub(r"[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣!.?]+", r" ", sent)
  sent = re.sub(r"\s+", " ", sent)
  return sent

def eng_preprocess(sent):
  sent = re.sub(r"([?.!,¿])", r" \1", sent)
  sent = re.sub(r"[^0-9a-zA-Z!.?]+", r" ", sent)
  sent = re.sub(r"\s+", " ", sent)
  return sent
```

 3. 데이터셋
   
 데이터셋 | 데이터 갯수 | 
 :-------:|:-----------:|
 Train Data | 180,000 |        
 Validation Data | 20,000 |
 Total Data | 200,000 |
 
 4. Seq2Seq with Attention 모델 핸들링
 
   * Attention
   
    RNN의 고질적인 문제인 Long-term problem로 인해 Attention 사용 
    ==> 입력 시퀀스를 동일한 비중으로 참고하는 것이 아닌, 예측 단어와 관련이 있는 입력 단어에 더욱 치중 ( Seq2Seq(LSTM) => Seq2Seq with Attention 변경) 
    
   * Bidirectional RNN(BiRNN)
   
    두개의 순환층을 사용하여 하나는 앞에서부터, 하나는 뒤에서부터 단어를 읽어 각 time step마다 두 출력을 연결해 prediction을 출력하는 방법 사용 ( 이전 데이터와의 관계 + 이후 데이터와의 관계 모두 학습) 
   
   * Add LSTM layers
    
    2층, 3층, 4층, 5층 비교
   
   * Optimizer
   
    Adam, Adadelta, AdamW 비교
   
   * Num of hidden units
   
    256, 512, 1024 비교
   
   * Embedding Dimension 
   
    64, 256, 512 비교 
    
   * Dropout 0.2
   
    셀 입력에 dropout 적용 
   
   * Recurrent Dropout 0.2 
    
    현재 input에 영향을 받는 parameter에만 dropout 적용 
   
   
   5. Transformer 
   
   * RNN 구조 대신 순서정보를 담기 위해서 Positional Encoding 단계 추가
    
    ( Embedding vector + Positional Encoding vector = Enbedding vector 생성 with 위치정보)
   * Self-Attention 사용 
    
    각 단어에 대해 나머지 단어들이 얼마나 유사한 지를 알려준다. 
    Q , K, V => 하나의 단어 Embedding vector x 가중치 행렬 
      1. Multi-Head Attention ( single * n개) 
      : 서로 다른 관점에서 문장 내 각각의 단어 유사도 정보를 뽑아내기 위해서 
      2. Masked Multi-Head Attention 
      : Decoder에서는 다음 단어 예측을 위해 전체문장을 가지고 오지 않기 위해서 
      3. Add & Norm 
      : 잔여학습을 위해서 
       
   

## 결과
- Attetion > Attention + BiLSTM > Attention + BiLSTM + LSTM layers
![image](https://user-images.githubusercontent.com/112064534/207072019-cd507ab8-478c-4dc1-b892-aa470b86a0c0.png)

- Attention + BiLSTM + LSTM layers > Attention + BiLSTM + LSTM 3층 + Dropout 
![image](https://user-images.githubusercontent.com/112064534/207072109-0bd104da-9d72-494c-be8a-b963ae1485b8.png)

==> Best Model 
* Mecab 
* Embedding Dimension = 512
* LSTM Layers = 3층 
* Dropout 0.2 
* Hidden units = 256

형태소 분석기는 5가지 중에서 Mecab을 이용하고, Embedding Dimension은 높으면 높을수록, LSTM Layers는 논문에 나온 4층보다는 3층이, Dropout은 0.2 , Recurrent Dropout은 사용하지 않는 것이, Hidden Units은 낮을수록 성능이 더 좋았다. 

하지만, 결과적으로 Transformer가 BiLSTM보다 Accuracy가 낮았음에도 더 좋은 성능을 보여주었는데 
첫 번째, LSTM은 Input을 순차적으로 받아 병렬처리가 어려운 반면, Transformer는 Sequence를 한번에 넣음으로써 병렬처리가 가능해졌기 때문이고
두 번째, LSTM의 Attention은 Encoder의 Hidden state와 Decoder의 Hidden state를 연산하여 Attention score를 계산하는 반면, Transformer의 Self-Attention은 Encoder로 들어간 벡터와 Encoder로 들어간 모든 벡터를 연산해서 Attention score를 계산함으로써 각 input data들의 유사도를 추정할 수 있기 때문에 
성능이 더 높은 이유이다. 

    


## 참조
-Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." Advances in neural information processing systems 27 (2014).
https://doi.org/10.48550/arXiv.1409.3215
-Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio. "Neural machine translation by jointly learning to align and translate." Accepted at ICLR 2015 as oral presentation 
https://doi.org/10.48550/arXiv.1409.0473
-VASWANI, Ashish, et al. Attention is all you need. Advances in neural information processing systems, 2017, 30.




