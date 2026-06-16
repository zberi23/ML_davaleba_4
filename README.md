# ML Davaleba 4 — FER2013 Facial Expression Recognition

[Kaggle Challenge: Challenges in Representation Learning](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge) — 7-კლასიანი emotion recognition 48×48 grayscale face images-ზე.

WandB project: https://wandb.ai/zberi23_ml/ml_davaleba_4

WandB Report: https://wandb.ai/zberi23_ml/ml_davaleba_4/reports/FER2013-Facial-Expression-Recognition-Iterative-Architecture-Comparison--VmlldzoxNzI0NzkyOA

## TL;DR

5 არქიტექტურა იტერაციულად, TinyCNN-დან ResNet18-მდე. ყოველ ნაბიჯზე specific პრობლემას ვაგვარებთ.

| # | Architecture | Test Acc | Macro F1 | Train-Val gap | რას ვაჩვენებთ |
|---|---|---|---|---|---|
| 1 | TinyCNN | 45.22% | 0.40 | small | underfit, capacity ცოტაა |
| 2 | SmallCNN | 56.45% | 0.53 | 43% | overfit, regularization-ის გარეშე |
| 3 | MediumCNN + BN + Dropout + L2 | 60.69% | 0.59 | 40% | regularization მარტო ვერ აგვარებს |
| 4 | VGG-mini + Data Augmentation | 70.47% | 0.69 | 6% | augmentation ჭრის overfit-ს |
| 5 | ResNet18 transfer learning | 71.69% | 0.71 | 25% | pretrained features |

ფინალური შედეგი 71.69% test accuracy, 0.71 macro F1 — human-level (~65-72%).

## Setup

Colab-ში Drive mount, ფოლდერი Drive-ში MyDrive/ML_davaleba_4/, data ფოლდერში fer2013_processed.npz უნდა იყოს (notebook 01 ქმნის). WandB API key Colab Secrets-ში WANDB_API_KEY სახელით.

## ფოლდერი

```
ML_davaleba_4/
├── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb      data download + EDA + train/val/test split
│   ├── 02_baseline_tiny_cnn.ipynb     TinyCNN — underfit demo
│   ├── 03_small_cnn.ipynb             SmallCNN — overfit demo
│   ├── 04_medium_cnn_batchnorm.ipynb  MediumCNN + BN + Dropout + L2
│   ├── 05_vgg_style.ipynb             VGG-mini + Data Augmentation
│   └── 06_resnet_transfer.ipynb       ResNet18 transfer learning
├── src/
│   └── utils.py                       data loading, training loop, WandB integration
├── data/                              (gitignored, npz იქ)
└── models/                            (gitignored, .pt checkpoints)
```

## იტერაციები — რა და რატომ

### Notebook 01 — Data Exploration

FER2013 ჩამოვტვირთე Kaggle API-ით. რაც აღმოვაჩინე: 35,887 image, train 28,709, val 3,589, test 3,589 (PublicTest/PrivateTest split). 7 ემოცია (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). Dataset მაგრად imbalanced — Disgust მხოლოდ 1.5%, Happy 25%. Pixel mean 129.47, std 65.03 (after /255: mean 0.51, std 0.26). npz-ში შევინახე — შემდეგ notebooks-ში სწრაფი load-ისთვის.

### Notebook 02 — TinyCNN, underfit demonstration

რა გავაკეთე: 1 conv layer, 8 filters, flatten, linear. სულ ~32K params.

რატომ: მინდოდა მენახა რა ხდება როცა capacity-ი ცოტაა. ლექტორმა გვითხრა "underfit/overfit მოდელების ჩვენებაც მნიშვნელოვანია" — ეს უმარტივესი დემო.

3 hyperparameter run:
- SGD lr=1e-3, val 39.96% (ნელი convergence)
- Adam lr=1e-3, val 44.78% (საუკეთესო)
- Adam lr=1e-2, val 40.99% (lr-ი მაგარია, instability ჩანდა val_loss-ში)

Test acc 45.22%, Macro F1 0.40.

Sanity checks ყველგან: forward loss ≈ log(7)=1.95, single batch overfit 100% accuracy-ს მიაღწია — ანუ model-ი მუშაობს ნორმალურად, უბრალოდ capacity-ი ცოტაა.

მთავარი დაკვირვება: train_acc 51%, val 45% — ორივე დაბალია, gap მცირე (~7%). ეს არ არის overfit, model-ი უბრალოდ ვერ სწავლობს მეტს. Disgust class-ი თითქმის უხილავია (recall 0.09 — model-ი მაგრად ფრთხილია, თუ პროგნოზდება უმეტესად სწორად, მაგრამ თითქმის არ პროგნოზდება).

გადაწყვეტილება: capacity გავზარდო.

### Notebook 03 — SmallCNN, overfit demonstration

რა გავაკეთე: 3 conv blocks (32 → 64 → 128 filters), MaxPool ყოველი ბლოკის ბოლოს, FC(4608→128→7). სულ ~683K params.

რატომ: TinyCNN-ის შემდეგ ლოგიკურია depth + filters გავზარდო. მინდოდა მენახა რა ხდება როცა model-ი მაგარია მაგრამ regularization-ი არ აქვს.

3 runs:
- 15 ep baseline, val 56.98%
- 30 ep no-reg, val 56.98% peak, ბოლოს train 96.4% vs val 53.9%, gap 42.6% — ნამდვილი overfit დემო
- 20 ep + L2 wd=1e-4, val 57.04%, test 56.45% — საუკეთესო

Test acc 56.45% (+11pp from TinyCNN), Macro F1 0.53.

მთავარი დაკვირვება: Run B-ში val_loss 1.2-დან 3.7-მდე გაიზარდა (3-ჯერ!), train_acc 96%-ს მიაღწია. ეს ნათელი overfit-ი — model-მა training data დაიმახსოვრა, generalization აღარ ხდება. Disgust აქ უკვე F1=0.47 (TinyCNN-ში იყო 0.16) — capacity-მა დაეხმარა. L2 weight decay ცოტათი ეხმარება, მაგრამ gap 35%-მდე მაინც ხდება.

გადაწყვეტილება: მძლავრი regularization მჭირდება — BatchNorm + Dropout.

### Notebook 04 — MediumCNN + BN + Dropout + L2

რა გავაკეთე: SmallCNN სტრუქტურა, ემატება BatchNorm თითო conv-ის შემდეგ + Dropout(0.5) FC-ში. სულ ~1.27M params.

რატომ: BN სტაბილიზებს გრადიენტებს და ფაქტობრივი regularization-ია, Dropout neuron-ების შემთხვევითი deactivation. L2 weight decay-ც დავამატე.

3 runs:
- BN only, val 58.48%, gap 40.5%
- BN + Dropout 0.3, val 58.96%, gap 40.2%
- BN + Dropout 0.5 + L2 + CosineAnnealingLR, val 59.77%, test 60.69% — საუკეთესო

Test acc 60.69% (+4pp), Macro F1 0.59.

მთავარი დაკვირვება (მოულოდნელი): ყველა 3 run-ში gap თითქმის ერთიდაიგივეა, ~40%. BN-მა optimization გააძლიერა, model-ი ახლა train data 99.6%-ით memorize-ს, val 60%-ზე გაიქცა. Regularization techniques ცალკე ვერ აგვარებენ პრობლემას ამ capacity-ის model-ისთვის.

გადაწყვეტილება: model-ი ძალიან ძლიერია train data-სთვის. გვჭირდება data augmentation რომ train data variability გავზარდოთ.

### Notebook 05 — VGG-mini + Data Augmentation

რა გავაკეთე: VGG-stilis 4 block-ი (პირველი 3-ში double conv: Conv→BN→ReLU×2 → MaxPool), 32→64→128→256 filters. სულ ~1.18M params.

Augmentation 3 დონის:
- mild: flip + RandomAffine(rot 10°, trans 5%)
- full: flip + RandomAffine(rot 15°, trans 10%, scale 0.9-1.1) + RandomCrop(48, padding=4)

3 runs:
- VGG no aug, val 63.44%, gap 36.5% (ისევ overfit)
- VGG mild aug, val 68.74%, gap 13.0%
- VGG full aug 50 ep, val 68.90%, test 70.47%, gap 6.2%

Test acc 70.47% (+10pp), Macro F1 0.69.

მთავარი დაკვირვება: augmentation იმუშავა მაგრად. Run A-ში train 99.8%, val 63% — overfit. Run B-ში train 81%, val 69% — augmentation train-ს უფრო hard-ს ხდის, val იზრდება. Run C-ში train 75%, val 69% — train ≈ val, generalization OK. Disgust F1 0.71-ი მიაღწია (TinyCNN-ში 0.16 იყო). Happy F1 0.88.

გადაწყვეტილება: from-scratch model 70%-ზე გავიდა. ვცადოთ transfer learning — pretrained features-ი იქნებ უფრო შორს წავიდეს.

### Notebook 06 — ResNet18 Transfer Learning

რა გავაკეთე: ResNet18 pretrained ImageNet-ზე. ბოლო FC layer შევცვალე (1000 → 7). Input pipeline: 48×48 grayscale → 224×224 RGB (resize + channel replication) + ImageNet normalization.

Progressive unfreezing strategy:
- Run A: frozen backbone, მხოლოდ FC trainable (linear probing) — 10 ep
- Run B: layer4 + FC unfrozen (partial fine-tune) — 12 ep
- Run C: full fine-tune, lr 1e-4 (low to preserve pretrained features) — 15 ep

Runs:
- A linear probe, val 43.38%, gap 1.2% (ImageNet features მარტო ცუდად მუშაობს faces-ისთვის)
- B partial, val 67.09%, gap 19.2%
- C full fine-tune, val 69.82%, test 71.69%, gap 25%

Test acc 71.69% (+1.2pp from VGG+aug), Macro F1 0.71.

მთავარი დაკვირვება: transfer learning იმუშავა, მაგრამ მოლოდინზე ნაკლები ეფექტი. მიზეზები: FER2013 grayscale 3-channel replication ნამდვილ RGB info-ს არ ემატება, ImageNet (natural images) ≠ FER2013 (faces) domain shift-ი, 48×48 → 224×224 upsample რეალურ info-ს არ ვამატებთ. Linear probing-მა მხოლოდ 43% აჩვენა — ცხადია რომ ImageNet features საკმარისად სპეციფიკური არ არის face-სთვის. მაგრამ როცა layer4 ვათავისუფლებთ, accuracy მაგრად ხტება (67%).

## საბოლოო Insights

Underfit → overfit → balanced path მუშაობს. TinyCNN ვერ სწავლობს, SmallCNN ძაან კარგად memorize-ს, MediumCNN BN+Dropout-ით ცოტა უკეთესია, VGG+Augmentation თვითონ generalization-ი.

Data augmentation > Regularization ამ task-ისთვის. BN/Dropout/L2 ერთად gap 40%-ით დატოვა, augmentation ერთი ნაბიჯით 6%-მდე ჩამოვიდა.

Class imbalance ნამდვილ პრობლემაა. Disgust 1.5%-ია data-ში, F1 0.16-დან 0.72-მდე გაიზარდა model capacity-ის და augmentation-ის გაუმჯობესებით.

Transfer learning კარგია, მაგრამ არა magic bullet. ResNet18 pretrained ImageNet-ზე მხოლოდ +1.2pp მოგვცა VGG+aug-ის შემდეგ. Grayscale faces ≠ ImageNet domain.

Forward/backward sanity checks ღირს. ყოველ notebook-ში ვამოწმე initial loss ≈ log(7) და single batch overfit. ეს ნიშნავს რომ model-ში არ არის გრადიენტული ან არქიტექტურული bug-ი.

## Reproducibility

ყოველ notebook-ში set_seed(42) გავუშვი — შედეგები ცოტა-ცოტა შესაძლოა იცვალოს GPU non-determinism-ის გამო, მაგრამ relative ranking იგივე.

Total compute ~3 საათი T4 GPU-ზე (notebook 05 და 06 ყველაზე გრძელი).
