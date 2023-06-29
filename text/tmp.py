"""
모델 학습

1. 인코더에 문장을 넣고 모든 출력과 최신 은닉 상태 추적하기
2. 디코더의 첫 번째 입력으로 <SOS> 토큰과 인코더의 마지막 은닉 상태가 첫번째 은닉 상태로 제공

간단한 if 문으로 teacher forcing 사용 유무 제어 가능
"""

teacher_forcing_ratio = 0.5

def train(
    input_tensor,
    target_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    max_length=MAX_LENGTH,
):
    encoder_hidden = encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
    loss = 0
    
    for ei in range(input_length):
        encoder_output, enocder_hidden = encoder(
            input_tensor[ei],
            encoder_hidden,
        )
        encoder_outputs[ei] = encoder_output[0, 0]
        
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # 목표를 다음 입력으로 전달하기
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, target_tensor[di])
            
            # teacher forcing
            decoder_input = target_tensor[di]
            
    else:
        # 모델의 예측을 다음 입력으로 사용
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(1)
            
            # 다음 입력으로 사용할 부분을 히스토리에서 분리
            decoder_input = topi.squeeze().detach()
            
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length