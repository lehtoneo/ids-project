import os
# This code extracts audio from each .mp4 file in a specified folder and stores them in ./audio (you need to create it manually)
##### You need to create the folder structure manually #####
# Paths
folder_name = "D:\\gitProjects\\ids-project\\data\\lockdown_math_announcement_480p_segments\\"

folder = os.fsencode(folder_name)
for file in os.listdir(folder):
    file_name = os.fsdecode(file)
    command = 'ffmpeg -i ' \
        + folder_name + file_name \
        + ' -f wav -ab 768000 -vn ' \
        + folder_name + '\\audio\\' + file_name.split('.')[0] + '_audio.wav'
    os.system(command)