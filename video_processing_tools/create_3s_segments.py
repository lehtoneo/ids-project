import os
# This code created three second segments from an input video and stores them in ...\data\{video_name}_segments\
##### You need to create the folder structure manually #####

# (Example) With these paths as specified you would need to create "D:\gitProjects\ids-project\data\lockdown_math_announcement_480p_segments\"

# Paths 
video_path = "D:\\gitProjects\\ids-project\\data\\"
video_name = "lockdown_math_announcement_480p.mp4"
output_path = video_path + video_name.split('.')[0] + "_segments\\"

start = 0
end = start + 3
while start < 60:
    # ffmpeg command to split the video to three second clip
    command = 'ffmpeg -i ' + video_path + video_name \
        + ' -ss ' \
        + str(start) + ' -to ' + str(end) \
        + ' ' \
        + output_path + video_name.split('.')[0] \
        + '_' + str(start) + '_' + str(end) + '.mp4'
        
    # Move the three second clip forward
    start +=3
    end = start + 3
    #print(command)
    # Execute the command
    os.system(command)