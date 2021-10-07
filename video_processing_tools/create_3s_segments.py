import os
# This code created three second segments from an input video and stores them in athe specified output path
##### You need to create the folder structure manually #####

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