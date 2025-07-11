import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os

# --- 1. 가상의 데이터 생성 ---
# 실제 데이터는 y = 2 * x + 3 에 약간의 노이즈를 더한 형태라고 가정
X_full = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
#Y_full = np.array([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]) + np.random.randn(len(X_full)) * 0.5
Y_full = np.array([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35])


print(f"X 값: {X_full}")
print(f"Y 값: {Y_full}")

# --- 2. 모델 매개변수 초기화 ---
w = 0.5 # 초기 가중치 (Weight)
b = 0.1 # 초기 편향 (Bias)
learning_rate = 0.01 # 학습률 (alpha)
epochs = 5 # 전체 데이터셋을 반복할 횟수
batch_size = 4 # 미니 배치 크기 (N)

# 각 단계별 w, b 저장용 리스트
w_history = []
b_history = []

print(f"초기 매개변수: w = {w:.4f}, b = {b:.4f}")
print("-" * 40)

# --- 3. 미니 배치 SGD 반복 ---
num_samples = len(X_full)
num_batches = num_samples // batch_size # 전체 배치의 수

for epoch in range(epochs):
    # 데이터를 무작위로 섞음 (매 에포크마다)
    permutation = np.random.permutation(num_samples)
    X_shuffled = X_full[permutation]
    Y_shuffled = Y_full[permutation]

    for i in range(num_batches):
        # 1. 미니 배치 샘플링
        # 현재 미니 배치의 시작과 끝 인덱스 계산
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        # 미니 배치 데이터 추출
        X_batch = X_shuffled[start_idx:end_idx]
        Y_batch = Y_shuffled[start_idx:end_idx]
        
        N = len(X_batch) # 현재 미니 배치에 포함된 샘플 수

        # 2. 미니 배치의 평균 손실에 대한 기울기 계산
        # 예측값 계산
        Y_pred_batch = w * X_batch + b
        
        # 오차 계산
        errors = Y_pred_batch - Y_batch
        
        # w에 대한 기울기 (dW) 계산
        dW = (2 / N) * np.sum(errors * X_batch)
        
        # b에 대한 기울기 (dB) 계산
        dB = (2 / N) * np.sum(errors)

        # 3. 매개변수 업데이트
        w -= learning_rate * dW
        b -= learning_rate * dB
        
        # 각 단계별 w, b 저장
        w_history.append(w)
        b_history.append(b)
        
        # 현재 손실 계산 (선택적)
        current_loss = np.mean(errors**2)

        print(f"  Epoch {epoch+1}/{epochs}, Batch {i+1}/{num_batches}:")
        print(f"    미니 배치 X: {X_batch}")
        print(f"    미니 배치 Y: {Y_batch}")
        print(f"    계산된 기울기: dW = {dW:.4f}, dB = {dB:.4f}")
        print(f"    업데이트된 매개변수: w = {w:.4f}, b = {b:.4f}, 손실 = {current_loss:.4f}\n")

print("-" * 40)
print(f"최종 매개변수: w = {w:.4f}, b = {b:.4f}")
# 실제 정답은 w=2, b=3에 가까워야 함을 기억하세요
print(f"구해진 회귀함수: y = {w:.4f} * x + {b:.4f}")

# --- 시각화: 각 단계별 회귀선 ---
plt.figure(figsize=(10, 6))
plt.scatter(X_full, Y_full, color='black', label='데이터')
x_line = np.linspace(X_full.min(), X_full.max(), 100)

# x, y축 범위 동일하게 설정
min_val = min(X_full.min(), Y_full.min())
max_val = max(X_full.max(), Y_full.max())
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

# 전체 단계 중 일부만(예: 10단계) 골라서 그리기 (너무 많으면 복잡하므로)
colors = cm.viridis(np.linspace(0, 1, len(w_history)))
step = max(1, len(w_history)//10)
for idx, i in enumerate(range(0, len(w_history), step)):
    y_line = w_history[i] * x_line + b_history[i]
    plt.plot(x_line, y_line, color=colors[i], alpha=0.8, linestyle='--', label=f'Step {i+1}')

# 마지막 회귀선은 진하게 표시
plt.plot(x_line, w * x_line + b, color='red', linewidth=2, label='최종 회귀선')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SGD 각 단계별 회귀선 변화')
plt.legend(loc='best', fontsize=9, ncol=2)
plt.show()

# --- 엑셀용: 각 단계별 예측값 CSV로 저장 ---
x_excel = X_full
step = max(1, len(w_history)//10)
y_preds = []
step_labels = []
for i in range(0, len(w_history), step):
    y_pred = w_history[i] * x_excel + b_history[i]
    y_preds.append(y_pred)
    step_labels.append(f'Step{i+1}')
# 마지막 단계(최종 회귀선)도 추가
final_y_pred = w * x_excel + b
y_preds.append(final_y_pred)
step_labels.append('Final')

# 데이터프레임 생성 및 저장
excel_df = pd.DataFrame({'x': x_excel})
for idx, label in enumerate(step_labels):
    excel_df[label] = y_preds[idx]
# 소스 파일과 같은 폴더에 저장
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'sgd_regression_steps.csv')
excel_df.to_csv(csv_path, index=False)
print(f'각 단계별 예측값이 {csv_path} 파일로 저장되었습니다.')