### Transformer 라이브러리 사용

모델 사용을 용이하게 하기 위해 `Transformers` 라이브러리가 만들어졌다
<br>
__목표__\
모델 적재, 학습, 저장하는 단일 API제공
<br>
__특징__
* 사용 용이성
* 유연성: 모든 모델이 `nn.Module` `tf.kerad.Model` 클래스로 표현됨
* __단순성__

<br>

__학습 목적__
* model, tokenizer를 함께 사용하는 end-to-end 예제 학습
* 모델 클래스 및 설정 클래스 톺아보기
* 모델을 로드하는 방법
* 모델이 예측을 출력하기 위해 수치적 입력을 처리하는 방법
* `tokenizer` API 학습