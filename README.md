# Data Augmentation
<h1>圖像去噪、輪廓增強</h1>

本文方法有CLAHEColor、NLM、Opening_operation、OTSU、Normalize、HSV、CLAHE+Normalize、HSV+CLAHE、NLM+Opening，主要針對圖像去噪及輪廓增強，程式呼叫方法如下所示。

<h2>1.CLAHEColor</h2>

- 參數說明

 CLAHE_Color：增強方法
 clahenum：設定每次處理的大小(值越大圖片顆粒感越重)

>>輸入範例：

```python=
 datagen=CustomDataGenerator(fun="CLAHE_Color",clahenum=40,dtype=int)
```
*下圖分別為clahenum所設定不同值的結果，由下圖所示clahenum設定為24輪廓較為明顯。

![](https://i.imgur.com/FV7kKRt.jpg)

<h2>2.NLM</h2>


- 參數說明
NLM：增強方法
h：決定濾波器強度。較高的值可以更好的消除噪聲，但也會刪除圖像細節
templateWindowSize：奇數(推薦值為7)
searchWindowSize：奇數(推薦值為21)

>>輸入範例：

```python=
datagen=CustomDataGenerator(fun="NLMOpening",h=10,templateWindowSize=7,searchWindowSize=21,dtype=int)
```
*下圖為NLM與原圖的對比，看起來更為平滑。
![](https://i.imgur.com/UI8Cdmq.png)

<h2>3.Opening_operation</h2>

:::info
- 參數說明
Opening_operation：增強方法
kernel：設定內核的大小
::: 
>>輸入範例：

```python=
datagen=CustomDataGenerator(fun="Opening_operation",kernel=5,dtype=int)
```
*下圖為Opening_operation與原圖的對比
![](https://i.imgur.com/NqBekq0.png)

<h2>4.OTSU</h2>

:::info
- 參數說明
OTSU：增強方法
::: 
>>輸入範例：

```python=
datagen=CustomDataGenerator(fun="OTSU",dtype=int)
```
*下圖為OTSU與原圖的對比
![](https://i.imgur.com/62RhjOZ.png)



<h2>5.Normalize</h2>

:::info
- 參數說明
normalize：增強方法
::: 
>>輸入範例：

```python=
datagen=CustomDataGenerator(fun="normalize",dtype=int)
```
*下圖為normalize與原圖的對比
![](https://i.imgur.com/5p77gRP.png)

<h2>6.HSV</h2>

:::info
- 參數說明
HSVcolor：增強方法
::: 
>>輸入範例：

```python=
datagen=CustomDataGenerator(fun="HSVcolor",dtype=int)
```
*下圖為normalize與原圖的對比
![](https://i.imgur.com/mWUZkj5.png)

<h2>7.CLAHE+Normalize</h2>

:::info
- 參數說明
CLAHENormalize：增強方法
clahenum：設定每次處理的大小
::: 
>>輸入範例：

```python=
datagen=CustomDataGenerator(fun="CLAHENormalize",clahenum=40,dtype=int)
```
*下圖為CLAHE-clahenum=40+normalize與原圖的對比
![](https://i.imgur.com/8Y9zGPM.png)

<h2>8.HSV+CLAHE</h2>

:::info
- 參數說明
CLAHEHSVcolor：增強方法
clahenum：設定每次處理的大小
::: 
>>輸入範例：

```python=
datagen=CustomDataGenerator(fun="CLAHEHSVcolor",clahenum=40,dtype=int)
```
*下圖為CLAHE-clahenum=40+HSV與原圖的對比
![](https://i.imgur.com/kpZ5Wkb.png)

<h2>9.NLM+Opening</h2>

:::info
- 參數說明
NLMOpening：增強方法
h：決定濾波器強度。較高的值可以更好的消除噪聲，但也會刪除圖像細節
templateWindowSize：奇數(推薦值為7)
searchWindowSize：奇數(推薦值為21)
kernel：設定內核的大小
::: 
>>輸入範例：

```python=
datagen=CustomDataGenerator(fun="NLMOpening",h=10,templateWindowSize=7,searchWindowSize=21,kernel=5,dtype=int)
```
*下圖為NLM+Opening與原圖的對比
![](https://i.imgur.com/urbKpZ7.png)

<h1>圖像去背</h1>

下方介紹圖像去背方法分別為，小波轉換、k-means


<h2>k-means</h2>

:::info
- 參數說明
kmeans：增強方法
num_clusters：獲得每個像素所屬的類別
::: 
>>輸入範例：

```python=
datagen=CustomDataGenerator(fun="kmeans",num_clusters=4,dtype=int)
```
*下圖為k-means與原圖的對比
![](https://i.imgur.com/g9ENDEA.png)



<h2>小波轉換</h2>

:::danger
該方法因為輸出為單通道，因此無法在本專案做使用。
在下方附上範例程式
::: 
>>輸入範例：

```python=
image = cv2.imread('03.png')
img = cv2.resize(img, (256, 256))
# 将多通道图像变为单通道图像
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# plt.figure('二维小波一级变换')
coeffs = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coeffs
# 将各个子图进行拼接，最后得到一张图
AH = np.concatenate([cA, cH], axis=1)
VD = np.concatenate([cV, cD], axis=1)
img2 = np.concatenate([AH, VD], axis=0)

cv2.imwrite("dwt.png",img2)
```
*下圖為小波轉換後的結果
![](https://i.imgur.com/Qcz8sYl.png)
