def compute_rms_per_frame_librosa(file_path, frame_duration_sec=2):
    """
    Computes the RMS value for each frame of a WAV audio file using Librosa and TensorFlow.
    
    Parameters:
    - file_path (str): Path to the WAV file.
    - frame_duration_sec (int, optional): Duration of each frame in seconds. Default is 2 seconds.
    
    Returns:
    - rms_np (np.ndarray): Array of RMS values for each frame.
    - sample_rate (int): Sample rate of the loaded audio file.
    - frame_length (int): Number of samples per frame.
    """
    # Enable device placement logging
     
    try:
        # Load the audio file using librosa with original sample rate
        waveform, sample_rate = librosa.load(file_path, sr=None, mono=True)
    except Exception as e:
        raise IOError(f"Error loading audio file '{file_path}': {e}")
    
    # Convert the waveform to a TensorFlow tensor
    waveform_tf = tf.constant(waveform, dtype=tf.float32)
    
    # Calculate frame parameters
    frame_length = int(frame_duration_sec * sample_rate)  # Number of samples per frame
    frame_step = frame_length  # No overlap
    
    # Ensure the signal length is sufficient for at least one frame
    num_samples = waveform_tf.shape[0]
    if num_samples < frame_length:
        raise ValueError(f"Audio file is too short for the desired frame duration of {frame_duration_sec} seconds.")
    
    # Frame the signal using TensorFlow's tf.signal.frame
    # This will create a 2D tensor where each row is a frame
    frames = tf.signal.frame(
        waveform_tf, 
        frame_length=frame_length, 
        frame_step=frame_step, 
        pad_end=False
    )
    
    # Calculate RMS for each frame
    # RMS = sqrt(mean(square(signal)))
    rms = tf.sqrt(tf.reduce_mean(tf.square(frames), axis=1))
    
    # Convert RMS Tensor to NumPy array
    rms_np = rms.numpy()
    
    return rms_np, sample_rate, frame_length
