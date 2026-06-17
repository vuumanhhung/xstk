# 🐍 Python cho Xác Suất Thống Kê – MA239
> Trường ĐH Thăng Long | Tổng hợp từ tài liệu của cô Nguyễn Thị Nhung

---

## 1. IMPORT CÁC THƯ VIỆN CẦN THIẾT

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy import stats
```

---

## 2. NUMPY – LÀM VIỆC VỚI MẢNG SỐ

### Tạo mảng
```python
arr = np.array([7, 8, 7, 10, 4, 3, 10, 9, 5, 8])
arr_2d = np.array([[7, 8, 7, 10, 4], [3, 10, 9, 5, 8]])
```

### Truy cập & lọc
```python
arr[0]           # phần tử đầu
arr[-1]          # phần tử cuối
arr[0:5]         # 5 phần tử đầu
arr[arr > 5]     # lọc phần tử > 5
np.where(arr == 10)  # tìm chỉ số của giá trị 10
```

### Thống kê mô tả bằng NumPy
```python
np.min(arr)           # giá trị nhỏ nhất
np.max(arr)           # giá trị lớn nhất
np.mean(arr)          # trung bình
np.median(arr)        # trung vị
np.var(arr)           # phương sai (mẫu: ddof=1)
np.std(arr)           # độ lệch chuẩn
np.percentile(arr, q=60)            # phân vị thứ 60
np.percentile(arr, q=[25, 50, 75])  # tứ phân vị
np.ptp(arr)           # khoảng biến thiên = max - min
np.sort(arr)          # sắp xếp tăng dần
```

---

## 3. PANDAS – LÀM VIỆC VỚI BẢNG DỮ LIỆU

### Đọc dữ liệu
```python
df = pd.read_csv('file.csv')
df = pd.read_table('file.txt', delimiter='\s+')
```

### Xem & truy cập
```python
df.head(5)          # 5 dòng đầu
df.shape            # (số dòng, số cột)
df.columns          # tên các cột
df.describe()       # tóm tắt thống kê
df['cot_A']         # truy cập một cột
df.loc[0]           # dòng đầu tiên
df.loc[[1, 3]]      # dòng 1 và 3
df.loc[:, 'cot_A']  # toàn bộ cột A
df.iloc[1, 1]       # dòng 1, cột 1 (theo vị trí)
```

### Lọc dữ liệu
```python
df[df['Age'] > 30]                        # tuổi > 30
df[(df['Gender'] == 'F') & (df['Age'] > 30)]  # kết hợp điều kiện
df.query("Age > 30")                       # dùng query()
```

### Thống kê mô tả bằng Pandas
```python
df['cot'].mean()       # trung bình
df['cot'].median()     # trung vị
df['cot'].mode()       # mode
df['cot'].var()        # phương sai
df['cot'].std()        # độ lệch chuẩn
df['cot'].skew()       # hệ số bất đối xứng (skewness)
df['cot'].kurtosis()   # hệ số đo độ nhọn (kurtosis)
df['cot'].quantile([0.25, 0.5, 0.75])  # tứ phân vị
df['cot'].describe()   # min, max, mean, std, Q1, Q2, Q3
```

### Bảng tần số
```python
df['cot'].value_counts()               # tần số
df['cot'].value_counts(normalize=True) # tần suất
pd.crosstab(df['GioiTinh'], df['HocLuc'])  # bảng chéo
pd.cut(df['Height'], bins=6)           # phân tổ tự động
pd.cut(df['Score'], bins=[0, 5, 7, 9, 10], include_lowest=True)  # phân tổ tự định
```

---

## 4. SCIPY.STATS – PHÂN PHỐI XÁC SUẤT & KIỂM ĐỊNH

### Phân phối nhị thức – Binomial(n, p)
```python
from scipy.stats import binom

n, p = 10, 0.3
# P(X = k)
binom.pmf(k=3, n=n, p=p)

# P(X <= k)
binom.cdf(k=3, n=n, p=p)

# P(X > k) = 1 - P(X <= k)
1 - binom.cdf(k=3, n=n, p=p)

# Kỳ vọng và phương sai
mean, var = binom.stats(n=n, p=p)
```

### Phân phối Poisson – Poisson(λ)
```python
from scipy.stats import poisson

lam = 5  # lambda
poisson.pmf(k=3, mu=lam)      # P(X = 3)
poisson.cdf(k=3, mu=lam)      # P(X <= 3)
1 - poisson.cdf(k=2, mu=lam)  # P(X > 2) = P(X >= 3)
```

### Phân phối chuẩn – Normal(μ, σ)
```python
from scipy.stats import norm

mu, sigma = 0, 1  # chuẩn tắc N(0,1)

# P(X <= x) = Φ(x)
norm.cdf(x=1.96, loc=mu, scale=sigma)

# P(X > x) = 1 - Φ(x)
1 - norm.cdf(x=1.96)

# P(a < X < b)
norm.cdf(b) - norm.cdf(a)

# Phân vị: tìm x sao cho P(X <= x) = p
norm.ppf(0.975)  # = 1.96 (phân vị 97.5%)

# Với phân phối chuẩn N(mu, sigma)
norm.cdf(x=75, loc=70, scale=10)
norm.ppf(0.95, loc=70, scale=10)
```

### Phân phối mũ – Exponential(λ)
```python
from scipy.stats import expon

lam = 2  # tỉ lệ
scale = 1/lam  # scipy dùng scale = 1/lambda

expon.cdf(x=1, scale=scale)    # P(X <= 1)
expon.pdf(x=1, scale=scale)    # hàm mật độ tại x=1
expon.ppf(0.9, scale=scale)    # phân vị thứ 90
```

### Phân phối t-Student
```python
from scipy.stats import t

df = 20  # bậc tự do
t.ppf(0.975, df=df)    # t(α/2) với α=0.05
t.cdf(x=2.0, df=df)    # P(T <= 2.0)
```

### Phân phối Chi-bình phương & F
```python
from scipy.stats import chi2, f

chi2.ppf(0.95, df=10)     # giá trị tới hạn χ²
f.ppf(0.95, dfn=5, dfd=20)  # giá trị tới hạn F
```

---

## 5. ƯỚC LƯỢNG KHOẢNG TIN CẬY

### Khoảng tin cậy cho trung bình (σ biết – dùng Z)
```python
from scipy import stats
import numpy as np

data = np.array([...])
n = len(data)
xbar = np.mean(data)
sigma = 5  # độ lệch chuẩn tổng thể đã biết
alpha = 0.05

z = stats.norm.ppf(1 - alpha/2)  # z(α/2)
margin = z * sigma / np.sqrt(n)
ci = (xbar - margin, xbar + margin)
print(f"Khoảng tin cậy {(1-alpha)*100}%: ({ci[0]:.4f}, {ci[1]:.4f})")
```

### Khoảng tin cậy cho trung bình (σ không biết – dùng t)
```python
from scipy import stats
import numpy as np

data = np.array([...])
n = len(data)
xbar = np.mean(data)
s = np.std(data, ddof=1)  # độ lệch chuẩn mẫu
alpha = 0.05

t_val = stats.t.ppf(1 - alpha/2, df=n-1)
margin = t_val * s / np.sqrt(n)
ci = (xbar - margin, xbar + margin)
print(f"Khoảng tin cậy {(1-alpha)*100}%: ({ci[0]:.4f}, {ci[1]:.4f})")

# Hoặc dùng hàm interval() trực tiếp:
ci2 = stats.t.interval(1-alpha, df=n-1, loc=xbar, scale=s/np.sqrt(n))
```

### Khoảng tin cậy cho tỷ lệ
```python
from scipy.stats import norm
import numpy as np

n = 100
p_hat = 0.6  # tỷ lệ mẫu
alpha = 0.05

z = norm.ppf(1 - alpha/2)
margin = z * np.sqrt(p_hat * (1 - p_hat) / n)
ci = (p_hat - margin, p_hat + margin)
print(f"Khoảng tin cậy cho p: ({ci[0]:.4f}, {ci[1]:.4f})")
```

---

## 6. KIỂM ĐỊNH GIẢ THUYẾT

### Kiểm định trung bình một tổng thể (t-test)
```python
from scipy.stats import ttest_1samp

data = [...]
mu0 = 50   # giá trị kiểm định H0: mu = mu0

# Hai phía: H1: mu ≠ mu0
t_stat, p_value = ttest_1samp(data, popmean=mu0)
print(f"t = {t_stat:.4f}, p-value = {p_value:.4f}")

# Một phía: H1: mu > mu0
p_one_tail = p_value / 2 if t_stat > 0 else 1 - p_value / 2

# Kết luận (α = 0.05)
alpha = 0.05
if p_value < alpha:
    print("Bác bỏ H0")
else:
    print("Chưa đủ cơ sở bác bỏ H0")
```

### Kiểm định tỷ lệ một tổng thể
```python
from statsmodels.stats.proportion import proportions_ztest

count = 60   # số lần thành công
nobs = 100   # cỡ mẫu
p0 = 0.5     # tỷ lệ kiểm định H0: p = p0

z_stat, p_value = proportions_ztest(count, nobs, value=p0)
print(f"z = {z_stat:.4f}, p-value = {p_value:.4f}")
```

### Kiểm định phương sai (Chi-bình phương)
```python
from scipy.stats import chi2
import numpy as np

data = [...]
sigma0_sq = 25   # phương sai kiểm định
n = len(data)
s_sq = np.var(data, ddof=1)

chi2_stat = (n - 1) * s_sq / sigma0_sq
p_value = 1 - chi2.cdf(chi2_stat, df=n-1)  # kiểm định phải
# hoặc 2 * min(chi2.cdf(...), 1 - chi2.cdf(...)) cho hai phía
print(f"χ² = {chi2_stat:.4f}")
```

---

## 7. VẼ BIỂU ĐỒ

### Histogram (phân phối tần số)
```python
plt.hist(data, bins=10, color='skyblue', edgecolor='black')
plt.title('Biểu đồ phân phối')
plt.xlabel('Giá trị')
plt.ylabel('Tần số')
plt.show()
```

### Boxplot (hộp và râu)
```python
plt.boxplot(data, vert=True, patch_artist=True)
plt.title('Biểu đồ hộp')
plt.show()
```

### Scatter (tán xạ)
```python
plt.scatter(x, y, color='blue', alpha=0.7)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Biểu đồ tán xạ')
plt.show()
```

### Bar chart
```python
plt.bar(categories, values, color='steelblue')
plt.title('Biểu đồ thanh')
plt.show()
```

---

## 8. HỒI QUY TUYẾN TÍNH

```python
from scipy.stats import linregress
import numpy as np

x = np.array([...])
y = np.array([...])

slope, intercept, r_value, p_value, std_err = linregress(x, y)

print(f"Hệ số góc b1 = {slope:.4f}")
print(f"Hệ số chặn b0 = {intercept:.4f}")
print(f"R² = {r_value**2:.4f}")
print(f"p-value = {p_value:.4f}")

# Dự đoán
y_pred = slope * x + intercept
```

---

## 9. THỐNG KÊ MÔ TẢ TỔNG HỢP (SCIPY.STATS)

```python
import scipy.stats as ss
import pandas as pd

df = pd.read_csv('data.csv')

# Mô tả tổng quát
ss.describe(df['cot'])
# → nobs, minmax, mean, variance, skewness, kurtosis

ss.mode(df['cot'])        # mode
ss.skew(df['cot'])        # skewness
ss.kurtosis(df['cot'])    # kurtosis
ss.variation(df['cot'])   # hệ số biến thiên = std/mean
ss.iqr(df['cot'])         # độ trải giữa (IQR = Q3 - Q1)
ss.sem(df['cot'])         # sai số chuẩn của trung bình
```

---

## 10. CÁC GIÁ TRỊ TỚI HẠN THƯỜNG GẶP

```python
from scipy.stats import norm, t, chi2, f

# Phân phối chuẩn (z)
norm.ppf(0.975)   # z(0.025) = 1.96
norm.ppf(0.95)    # z(0.05)  = 1.645
norm.ppf(0.99)    # z(0.01)  = 2.326

# Phân phối t (cần biết bậc tự do df)
t.ppf(0.975, df=20)   # t(0.025; 20)
t.ppf(0.95, df=10)    # t(0.05; 10)

# Chi-bình phương
chi2.ppf(0.95, df=10)    # χ²(0.05; 10)
chi2.ppf(0.025, df=10)   # χ²(0.975; 10)

# Phân phối F
f.ppf(0.95, dfn=3, dfd=20)  # F(0.05; 3, 20)
```

---

## GHI CHÚ QUAN TRỌNG

| Lưu ý | Chi tiết |
|-------|----------|
| `ddof=1` | Dùng khi tính std/var của **mẫu** (n-1), không dùng với tổng thể |
| `scale = 1/λ` | SciPy dùng `scale` thay vì `λ` cho phân phối mũ |
| `loc=mu, scale=sigma` | Tham số cho phân phối chuẩn trong scipy |
| `p-value < α` | → Bác bỏ H₀ |
| `p-value ≥ α` | → Chưa đủ cơ sở bác bỏ H₀ |
