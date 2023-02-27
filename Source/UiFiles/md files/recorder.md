# Recorder
Recorder can either create photos or videos of full grid. To create a photo just click on "Create image".
To make a video follow these steps:
1. Click **"New recording"**
2. Click **"Start recording"**
3. Record for however long you want.
4. Click **"Pause recording"**
5. Click **"Compile intermediate data into video"**
6. Wait until video is compiled. Finished video will be in the "path to program dir/videos". The video will be named with the timestamp of the start of the recording

You can compile several videos in parallel, but it is not recommended.

### Various setting.
* **"Grid tbuffer size"** - The size of a tbuffer. The program will first write grid states to the tbuffer, before writing all at once to the drive.
* **"Log every n tick"** - Will log every n tick.
* **"Video output FPS"** - FPS that will be set during video construction.

### Various buttons.
* **"New recording"** - Will create new folder in /temp/ with timestamp where recording will be stored.
* **"Stop recording"** - Will stop the recording, flushing data in the tbuffer to the disk.
* **"Clear intermediate data"** - Will stop the recording before freeing the tbuffer space.
* **"Delete all intermediate data from disk"** - Will delete everything in the /temp/ folder.
* **"Compile intermediate data into video"** - The output will be in /videos/ folder. Compilation is done in two stages:
* **"Load intermediate data location"** - Choose the folder with the recording.
