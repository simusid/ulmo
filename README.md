# ulmo
Large acoustic models

# Things to investigate
- alternatives to K-means and better tokenization
  - Self supervised pretext tasks such as next frame prediction, infill, ABX (STFT, Mel Spec, CQT, CWT, DWT, WPT choice tasks)
- long context to discover repeated impulses
- adding an \<UNK\> token 
- Try a denoising autoencoder in place of kmeans to create labels.
- Data augmentation by using sets of spectrogram parameters (vary SR +- 10%)
- methods of self supervision such as SimCLR, wav2vec2,and BYOL
