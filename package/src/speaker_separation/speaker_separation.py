import torch
import scipy
import scipy.io.wavfile
import os
import numpy as np
try:
    from functions import load_model, read_wav
except:
    from speaker_separation.functions import load_model, read_wav

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = {'AIcloud': '/home/jovyan/.local',
        'Staging': './env',
        'Uat': './env',
        'Prod': './env'}
path = '/lib/python3.6/site-packages/speaker_separation/dprnn_model'



class Speaker_separator():
    def __init__(self, model_path='', device=device, env_now='AIcloud'):
        self.load_dprnn_model(model_path=model_path, device=device, env_now=env_now)

    def load_dprnn_model(self, model_path='', device=device, env_now='AIcloud'):
        if model_path!='':
            self.model = load_model(model_path, device)
        else:
            m_path = root[env_now]+path
            self.model = load_model(m_path, device)

    def separate_by_dprnn(self, wav_array_list=[], wav_path_list=[],
                                save_path=os.getcwd(), save_name=[],
                                input_array=True, input_path=False, return_array=True):
        """
        Speaker separator with DPRNN model.

        Arguments
        ---------
                wav_array_list: All wav data(s) which you want to separate.
                                type: list. All data(s) in this list must be 1 x n np.array.
                                example: [array([[0., 0., ..., 0.]], dtype=float32),
                                          array([[0., 1., ..., 0.]], dtype=float32)]
                wav_path_list: All wav path(s) which you want to separate.
                               type: list. All path(s) in this list must be str.
                               example: ['/home/jovyan/my_project/aaa.wav',
                                         '/home/jovyan/my_project/bbb.wav']
                save_path: If you want to let output be wav file, which path you want to save its.
                           type: str. example: '/home/jovyan/my_project/result'
                save_name: Wav name(s) for your output wav file(s).
                           type: list. All name(s) in this list must be str.
                           example: ['aaa_result', 'bbb_result']
                input_array: Is your input np.array?
                input_path: Is your input wav path?
                            input_array and input_path please choose one be True.
                return_array: Do you want to let output be np.array?
                              If not, it will be saved by wav file in save_path.

        Returns
        -------
                if return_array: type: list. All data(s) in this list must be 2 x n np.array.
                                 example: [array([[0., 0., ..., 0.],
                                                  [1., 1., ..., 0.]], dtype=float32),
                                           array([[0., 1., ..., 0.],
                                                  [1., 0., ..., 1.]], dtype=float32)]
                                 If something wrong, you will get list of message.
                                 example: ['Sorry, wav_array_list[1] is failed.']
                not return_array: You will get output wav files in save_path.
                                  example: '/home/jovyan/my_project/result/aaa_s1.wav'
                                           '/home/jovyan/my_project/result/aaa_s2.wav'
                                  If something wrong, you will get list of message.
                                  example: ['Sorry, separate wav_path_list[1] is failed.']
        """
        assert (input_array+input_path)==1, 'Please choose one of your input data type, numpy array or wav path.'

        result_array = []
        if input_array:
            assert len(wav_array_list) > 0, 'Please put wav data(s) in wav_array_list.'
            assert sum([type(i) == type(np.array([])) for i in wav_array_list]) == len(wav_array_list), 'All datas in wav_array_list must be numpy array.'
            assert sum([i.shape[0] == 1 for i in wav_array_list]) == len(wav_array_list), 'All arrays shape must be 1 x n.'
            for i in range(len(wav_array_list)):
                try:
                    result_array.append(self.model.separate_wav(wav_array_list[i]))
                except:
                    result_array.append(f'Sorry, wav_array_list[{i}] is fail.')
        if input_path:
            assert len(wav_path_list) > 0, 'Please put wav path(s) in wav_path_list.'
            for i in range(len(wav_path_list)):
                try:
                    result_array.append(self.model.separate_file(wav_path_list[i]))
                except:
                    result_array.append(f'Sorry, separate wav_path_list[{i}] is failed.')

        if return_array:
            return result_array

        assert len(save_name) == len(wav_array_list) or len(save_name) == len(wav_path_list) or save_name == [], 'Please check length of save_name.'
        assert sum([type(i) == str for i in save_name]) == len(save_name) or save_name == [], 'Please let all values in save_name be string.'
        save_output_result = []
        for i in range(len(result_array)):
            if type(result_array[i]) == str:
                save_output_result.append(f'Sorry, separate wav_path_list[{i}] is failed.')
            else:
                if save_name == []:
                    wav_name = f'output_wav_{i}'
                else:
                    wav_name = save_name[i]
                try:
                    scipy.io.wavfile.write(f"{save_path}/{wav_name}_s1.wav", 8000, result_array[i][0])
                    scipy.io.wavfile.write(f"{save_path}/{wav_name}_s2.wav", 8000, result_array[i][1])
                except:
                    save_output_result.append(f'Sorry, save wav_path_list[{i}] result is failed.')
        if save_output_result == []:
            return 'Separate all wavs successed.'
        else:
            return save_output_result

