# Image Segmentation : U-net

![2.png](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/2.png)

---

- 사전 배경
    - 이론적 배경 및 연구 목적
        - Convolution Neural Network는 이미 오랫동안 존재했지만, 사용 가능한 Train Sets의 크기와 Network의 크기로 인해 성공이 제한됨
        - 많은 Visual Tasks, 특히 Biomedical Image Processing에서 원하는 출력에는 Localization이 포함되어야 했음 ⇒ 즉, 클래스 레이블이 각 픽셀에 할당되어야 한다는 말임
        
        > **CNN보다 특징을 잘 잡아낼 수 있도록 경계 정보와 특징을 잘 보존하여 이미지의 
        문맥을 파악해야 하는 등 조금 더 높은 수준의 이미지 이해를 요구하기 위함**
        > 

---

- U-net 소개
    - 특징
        - 넓은 범위의 이미지 픽셀로부터 의미정보를 추출하고 해당 의미정보를 기반으로 
        각 픽셀마다 객체를 분류하는 U모양의 아키텍처
        - 근접한 객체 경계를 잘 구분하게 학습하기 위해 Weighted Loss(가중치 오차) 제시
        - Semantic Segmentation을 의료용 Biomedical(생체의학) Image 분석에 사용한 모델
    
    이미지를 **Pixel기반으로 분할**하는 알고리즘으로 Semantic Segmentation을 수행
    넓은 범위의 이미지 픽셀로 부터 의미 정보를 추출하고, **(수축경로)**
    의미 정보를 기반으로 각 픽셀마다 객체를 분류하는 기능 **(확장경로)**
    
    <aside>
    💡 **입력(Input)**은 이미지이며, **픽셀별 RGB**데이터이다.
    **출력(Output)**은 이미지이며, **픽셀별 객체 구분 정보(Class)**이다.
    
    </aside>
    

---

- **작동 원리** + **Related Work**
    - **Related Work**
        - Semantic Segmentation
            - Segmentation : 이미지에서 픽셀단위로 객체를 추출하는 방법
            - Semantic Segmentation은 Image Segmentation이라고도 하며, 실제로 인식할 수 있는 물리적 의미 단위로 인식하는 Segmentation을 의미함
        - Batch Normalization
            - Normalization
                
                ![optimum](https://user-images.githubusercontent.com/101788136/225791702-18455870-a9ba-482e-b373-ba00e3f4934c.png)
                
                                          (좌) Normalization 적용전 / (우) Normalization 적용후
                
                ---
                
                - 기본적으로 정규화를 하는 이유는 학습을 더 빨리 하기 위해서 또는 Local optimum문제에 빠지는 가능성을 줄이기 위해 사용됨
                - 정규화하여 그래프를 왼쪽에서 오른쪽으로 만들어, Local optimum에 빠질 수 있는 가능성을 낮춰주게 됨
                - 정규화를 사용하면 Gradient Vanishing/Exploding을 어느정도 방지할 수 있음.
            - Batch Normalization Background
                - 단순히 고전적인 정규화 방식인 **Whitening**만을 시키면 해당 과정과 파라미터를 계산하기 위한 최적화와 무관하게 진행되기 때문에 특정 파라미터가 계속 커지는 상태로 Whitening이 진행될 수 있어서 Batch Normalization이 등장했음
                - **Whitening**
                    
                    각 레이어의 입력의 분산을 평균 0, 표준편차 1인 입력값으로 정규화시키는 방법
                    
                    Internal Covariance Shift 때문에 학습에서 불안정화가 일어나 Whitening을 적용
                    
                    > Covariate Shift : 이전 레이어의 파라미터 변화로 인하여 현재 레이어의 입력의 분포가 바뀌는 현상
                    Internal Covariate Shift : 레이어를 통과할 때 마다 Covariate Shift가 일어나면서 입력의 분포가 약간씩 변하는 현상
                    > 
            - Batch Normalization : 각 레이어마다 정규화하는 레이어를 두어, 변형된 분포가 나오지 않도록 조절하며, 학습하는 과정 자체를 전체적으로 안정화하여 학습 속도를 가속 시킬 수 있는 근본적인 방법이다.
        - Upsampling / Downsampling
            
            ![2.png](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/2%201.png)
            
            - Upsampling : 해당 분류에 속하는 데이터가 적은 쪽을 표본으로 더 추출하는 방법 
            ⇒ 작은 사진을 크게 키우는 것
            - Downsampling : 데이터가 많은 쪽을 적게 추출하는 방법 
            ⇒ 큰 사진을 작게 줄이는 것
        - Skip Connection
            - layer의 output을 몇 개의 layer를 건너뛰고 다음 layer의 input에 추가하는 기법
    - **작동 원리**
        - **Data Preprocessing**
            - Overlap-tile strategy
                
                작은 영역을 예측하기 위해서 큰 영역을 학습한다고 이해하면 된다.
                겹치는 부분이 존재하도록 이미지를 자르고 Segmentation하기 때문에 Overlap Tile이라고 불린다.
                
                ![Untitled](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/Untitled.png)
                
                이미지의 크기가 큰 경우 이미지를 자른 후 각 이미지에 해당하는 Segmentation을 진행해야 한다. 
                U-Net은 Input과 Output의 이미지 크기가 다르기 때문에 위 그림에서 처럼 파란색 영역을 Input으로 넣으면 노란색 영역이 Output으로 추출된다. 
                동일하게 초록색 영역을 Segmentation하기 위해서는 **빨간색 영역을 모델의 Input으로 사용**해야 한다. 
                
                즉, **겹치는 부분이 존재**하도록 이미지를 자르고 Segmentation하기 때문에 Overlap Tile 전략이라고 논문에서는 지칭한다.
                
            - Mirroring Extrapolate
                
                ![Untitled](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/Untitled%201.png)
                
                ![Untitled](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/Untitled%202.png)
                
                이미지의 경계부분을 예측할 때에는 Padding을 넣어 활용하는 경우가 일반적이다. 본 논문에서는 이미지 경계에 위치한 이미지를 복사하고 좌우 반전을 
                통해 **Mirror 이미지를 생성한 후 원본 이미지의 주변에 붙여** Input으로 사용.
                
                > *본 논문의 실험분야인 biomedical 에서는 세포가 주로 등장하고, 세포는 상하 좌우 대칭구도를 이루는 경우가 많기 때문에 Mirroring 전략을 사용했을 것이라고 추측*
                > 
            - Data Augmentation
                
                Data의 양이 적으면 데이터 증강을 통해 풍부한 데이터로 모델이 단단해지도록 학습한다. 논문에서 사용한 데이터 증강으로는 Rotation, Shift, Elastic deformation 등이 있다.
                논문의 저자는 작은 데이터 셋을 가지고 segmentation network를 학습시킬 때 random elastic deformation이 key concept로 보인다고 하였다.
                
                - Random Elastic Deformation
                    
                    ![Untitled](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/Untitled%203.png)
                    
                    Biomedical 분야에서 다양한 Deformations 방법 중 Elastic Deformations 을 사용한 이유는 다음과 같다. 
                    
                    1) the small amount of available data
                    
                    2) class imbalance
                    
                    이와 같은 문제를 해결하기 위해 elastic transformation을 도입하나, elastic transformation 이외에도 다른 augmentation도 충분히 사용 가능하다.
                    
                    >  Elastic Deformation 사용을 추천하는 경우 : 연속체에서 어떤 힘이나 시간 흐름으로 인해 변화가 발생하는 경우. 이 힘이 제거된 후 변형이 원래처럼 돌아오게 되면 이 변형을 탄성이라고 합니다. 이렇게 탄성이 있는 경우는 같은 물체라 해도 촬영 방법이나 각도 등에 의해서 다른 결과를 가져올 수 있으므로 이럴 때 사용하면 좋다고 하나 이 외의 경우에도 사용 (사람의 글씨체 차이에 따른 MNIST 적용 등에도 사용 했었음) 가능합니다.
                    
                    ![Untitled](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/Untitled%204.png)
                    
        - **Model Structure**
            - 수축경로 (Contracting Path)
                
                
                ![images_minkyu4506_post_ed171591-5e89-4bc9-bab3-e3be41fb85ef_스크린샷 2021-08-31 오후 4.48.14.png](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/images_minkyu4506_post_ed171591-5e89-4bc9-bab3-e3be41fb85ef_%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2021-08-31_%25EC%2598%25A4%25ED%259B%2584_4.48.14.png)
                
                ![Untitled](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/Untitled%205.png)
                
                <aside>
                ⬇️ ***Down Sampling***
                **3x3 Conv + ReLu + BatchNormalization (NoPadding, Strides = 1)
                3x3 Conv + ReLu + BatchNormalization (NoPadding, Strides = 1)
                2x2 MaxPooling (Strides = 2)**
                
                </aside>
                
                Downsampling 과정을 반복하여 특징맵(Feature Map)을 생성함
                
                - 주변 픽셀들을 참조하는 범위(필터의 범위)를 넓혀가며 이미지로부터 **Contextual 정보(특징 정보)를 추출**하는 역할을 함
                - 처음 Input Channel을 제외하고 Downsampling 할 때마다 Channel의 수를 2배씩 증가시키면서 진행함
                    
                    $$
                    1→64→128→256→512→1024
                    $$
                    
                - **3×3 Convolution**는 패딩을 하지 않아 특징맵(Feature Map)의 크기가 감소한다.
                
                > *논문에서는 Batch-Normalization이 언급되지 않았으나 구현체 및 다수의 리뷰에서 Batch-Normalization을 사용하는 것을 확인할 수 있다. [[참고자료]](https://github.com/milesial/Pytorch-UNet)*
                > 
            - 전환구간 (Bottle Neck)
                
                <aside>
                ♾️ ***전환***
                **3x3 Conv + ReLu + BatchNormalization (NoPadding, Strides = 1)
                3x3 Conv + ReLu + BatchNormalization (NoPadding, Strides = 1)
                Dropout**
                
                </aside>
                
                마지막에 적용된 **Dropout Layer**는 모델을 **일반화하고 노이즈에 견고하게(Robust)**
                 **만들며, 과적합을 막아주는 장치**이다.
                
            - 확장경로 (Expanding Path)
                
                ![Untitled](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/Untitled%206.png)
                
                <aside>
                ⬆️ ***Up Sampling*
                2x2 DeConv (Strides = 2)**
                수축 경로에서 동일한 Level의 특징맵(Feature Map)을 추출하고 크기를 맞추기 위하여 자른 후(Cropping) 이전 Layer에서 생성된 특징맵(Feature Map)과 **연결(Concatenation)**합니다.
                crop을 하는 이유는 contracting path에 있는 feature map과 expansive path에 있는 feature map의 해상도가 같지 않기 때문입니다.
                **3x3 Conv + ReLu + BatchNormalization (NoPadding, Strides = 1)
                3x3 Conv + ReLu + BatchNormalization (NoPadding, Strides = 1)**
                
                </aside>
                
                확장경로는 수축 경로에서 생성된 **Contextual 정보와 위치정보 결합**하는 역할을 합니다.
                **(Kernel의 개수를 반으로 줄인 CNN에 적용하기 전에 반대쪽 contracting path에서 같은 층에 있는 feature map과 합칩니다.)**
                
                - **Upsampling 과정** 을 반복하여 특징맵(Feature Map)을 생성
                - 수축 경로에서 생성된 **Contextual 정보와 위치정보 결합**하는 역할
                - 동일한 Level에서의 수축경로의 특징맵과 확장경로의 특징맵의 크기가 다른 이유는 여러번의 패딩이 없는 **3×3 Convolution Layer**를 지나면서 특징맵의 크기가 줄어들기 때문
                
                CNN -> upsampling -> 맞은편 contracting path의 feature map을 copy and crop -> CNN ->…
                
                ![Untitled](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/Untitled%207.png)
                
                <aside>
                ➕ 확장경로의 마지막에는 Class의 갯수만큼 필터를 갖고 있는 
                **1×1 Convolution Layer**가 있습니다. 
                **1×1 Convolution Layer**를(Class개수 == 채널 개수) 통과한 후 각 픽셀이 어떤 Class에 해당하는지에 대한 정보를 나타내는 **3차원(Width × Height × Class) 벡터**가 생성
                
                ![images_minkyu4506_post_f2ecc5e0-0d5e-4677-a2d2-d0d40261e4a7_스크린샷 2021-08-30 오후 9.27.38.png](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/images_minkyu4506_post_f2ecc5e0-0d5e-4677-a2d2-d0d40261e4a7_%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2021-08-30_%25EC%2598%25A4%25ED%259B%2584_9.27.38.png)
                
                </aside>
                
                - Skip Connection을 통해 수축 경로에서 생성된 Contextual 정보와 위치정보를 결합하는 역할
        - **Weight Loss**
            
            ![boundary_target.png](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/boundary_target.png)
            
            - Biomedical 분야를 위한 모델이다 보니 위 이미지처럼 작은 경계를 분리할 수 있도록 학습되어야 함
            - 각 픽셀이 경계와 얼마나 가까운지에 따른 Weight-Map을 만들고 학습할 때 경계에 가까운 픽셀의 Loss를 Weight-Map에 비례하게 증가 시킴으로써 경계를 학습하도록 함
            - 픽셀과 경계의 거리가 가까우면 큰 값을 갖게 되므로 해당 픽셀의 Loss값이 커지게 됨
            - 수식
                
                energy function은 맨 마지막에 얻은 feature map에 픽셀 단위로 [soft-max](https://en.wikipedia.org/wiki/Softmax_function)를 수행하고 여기에 cross entropy loss function을 적용하는 식이라고 하는
                
                ![Untitled](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/Untitled%208.png)
                
                여기서의 x는 특징맵에 있는 각 픽셀을 말한다.
                
                w(x)는 weight map이라는 픽셀 별로 가중치를 부과하는 역할이다.
                
                즉, 한마디로 작은 경계를 잘 분리할 수 있도록 픽셀과 경계와의 거리에 따라서 가중치를 줘서 경계를 학습하도록 한다는 것이다.
                
                ![Untitled](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/Untitled%209.png)
                
                특정 클래스가 가지는 픽셀의 주파수의 차이를 보완해주는 식이다.
                
                d1 : 바로 가장 가까운 클래스의 테두리와의 거리
                d2 : 두 번째로 가까운 클래스의 테두리와의 거리
                E = weight map * log(픽셀에서 얻은 클래스별 예측값을 soft-max한 것)
                

---

- U-net 코드
    
    실제로 구현해보기 (모델 가져오기)
    
    @박민 확인해주세여 :)
    
    [U-net 실제 구현 코드](http://machinelearningkorea.com/2019/08/25/u-net-실제-구현-코드/)
    
    - Conv Block
        
        ![Untitled](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/Untitled%2010.png)
        
        ```python
        """ Conv Block """
        class ConvBlock(tf.keras.layers.Layer):
            def __init__(self, n_filters):
                super(ConvBlock, self).__init__()
        
                self.conv1 = Conv2D(n_filters, 3, padding='same')
                self.conv2 = Conv2D(n_filters, 3, padding='same')
        
                self.bn1 = BatchNormalization()
                self.bn2 = BatchNormalization()
        
                self.activation = Activation('relu')
        
            def call(self, inputs):
                x = self.conv1(inputs)
                x = self.bn1(x)
                x = self.activation(x)
        
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.activation(x)
        
                return x
        ```
        
    - Encoder Block
        
        ![Untitled](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/Untitled%2011.png)
        
        ```python
        """ Encoder Block """
        class EncoderBlock(tf.keras.layers.Layer):
            def __init__(self, n_filters):
                super(EncoderBlock, self).__init__()
        
                self.conv_blk = ConvBlock(n_filters)
                self.pool = MaxPooling2D((2,2))
        
            def call(self, inputs):
                x = self.conv_blk(inputs)
                p = self.pool(x)
                return x, p
        ```
        
    - Decoder Block
        
        ![Untitled](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/Untitled%2012.png)
        
        ```python
        """ Decoder Block """
        class DecoderBlock(tf.keras.layers.Layer):
            def __init__(self, n_filters):
                super(DecoderBlock, self).__init__()
        
                self.up = Conv2DTranspose(n_filters, (2,2), strides=2, padding='same')
                self.conv_blk = ConvBlock(n_filters)
        
            def call(self, inputs, skip):
                x = self.up(inputs)
                x = Concatenate()([x, skip])
                x = self.conv_blk(x)
        
                return x
        ```
        
    - **U-net 전체적인 모델링 (*unet_model.py*)**
        
        ```python
        # U-Net model
        # coded by st.watermelon
        
        import tensorflow as tf
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose
        from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate
        
        """ Conv Block """
        class ConvBlock(tf.keras.layers.Layer):
            def __init__(self, n_filters):
                super(ConvBlock, self).__init__()
        
                self.conv1 = Conv2D(n_filters, 3, padding='same')
                self.conv2 = Conv2D(n_filters, 3, padding='same')
        
                self.bn1 = BatchNormalization()
                self.bn2 = BatchNormalization()
        
                self.activation = Activation('relu')
        
            def call(self, inputs):
                x = self.conv1(inputs)
                x = self.bn1(x)
                x = self.activation(x)
        
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.activation(x)
        
                return x
        
        """ Encoder Block """
        class EncoderBlock(tf.keras.layers.Layer):
            def __init__(self, n_filters):
                super(EncoderBlock, self).__init__()
        
                self.conv_blk = ConvBlock(n_filters)
                self.pool = MaxPooling2D((2,2))
        
            def call(self, inputs):
                x = self.conv_blk(inputs)
                p = self.pool(x)
                return x, p
        
        """ Decoder Block """
        class DecoderBlock(tf.keras.layers.Layer):
            def __init__(self, n_filters):
                super(DecoderBlock, self).__init__()
        
                self.up = Conv2DTranspose(n_filters, (2,2), strides=2, padding='same')
                self.conv_blk = ConvBlock(n_filters)
        
            def call(self, inputs, skip):
                x = self.up(inputs)
                x = Concatenate()([x, skip])
                x = self.conv_blk(x)
        
                return x
        
        """ U-Net Model """
        class UNET(tf.keras.Model):
            def __init__(self, n_classes):
                super(UNET, self).__init__()
        
                # Encoder
                self.e1 = EncoderBlock(64)
                self.e2 = EncoderBlock(128)
                self.e3 = EncoderBlock(256)
                self.e4 = EncoderBlock(512)
        
                # Bridge
                self.b = ConvBlock(1024)
        
                # Decoder
                self.d1 = DecoderBlock(512)
                self.d2 = DecoderBlock(256)
                self.d3 = DecoderBlock(128)
                self.d4 = DecoderBlock(64)
        
                # Outputs
                if n_classes == 1:
                    activation = 'sigmoid'
                else:
                    activation = 'softmax'
        
                self.outputs = Conv2D(n_classes, 1, padding='same', activation=activation)
        
            def call(self, inputs):
                s1, p1 = self.e1(inputs)
                s2, p2 = self.e2(p1)
                s3, p3 = self.e3(p2)
                s4, p4 = self.e4(p3)
        
                b = self.b(p4)
        
                d1 = self.d1(b, s4)
                d2 = self.d2(d1, s3)
                d3 = self.d3(d2, s2)
                d4 = self.d4(d3, s1)
        
                outputs = self.outputs(d4)
        
                return outputs
        ```
        

---

- 요약
    
    1) Preprocessing
    
    - Overlap-tile strategy
    - Mirroring Extrapolate
    - Data Augumentaion
    
    2) Training
    
    - Contracting Path
    - Bottle Neck
    - Expanding Path
    - Weight Loss
    
    3) Output
    
    ![images_minkyu4506_post_f2ecc5e0-0d5e-4677-a2d2-d0d40261e4a7_스크린샷 2021-08-30 오후 9.27.38.png](Image%20Segmentation%20U-net%20f0b574dd914a46e1802c83d6ebbad9e8/images_minkyu4506_post_f2ecc5e0-0d5e-4677-a2d2-d0d40261e4a7_%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2021-08-30_%25EC%2598%25A4%25ED%259B%2584_9.27.38.png)
    
    $$
    Input( w*h*RGB ) \\→ Model \\→
    Output( w*h*class )
    $$
    
    **모델의 구조 이해했다면, Conv과정에서 Padding이 전혀 사용이 되지 않으므로, 
    모델의 출력 크기는 입력크기보다 작을 수 밖에 없다.**
    

---

- 참고 문헌
    - 블로그
        
        [[논문리뷰]U-Net - 새내기 코드 여행](https://joungheekim.github.io/2020/09/28/paper-review/)
        
        [U-Net 논문 리뷰 - U-Net: Convolutional Networks for Biomedical Image Segmentation](https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a)
        
    - 논문
        
        [https://arxiv.org/pdf/1505.04597.pdf](https://arxiv.org/pdf/1505.04597.pdf)
        

---
