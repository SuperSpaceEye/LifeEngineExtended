// Adapted from https://stackoverflow.com/questions/34511312/how-to-encode-a-video-from-several-images-generated-in-a-c-program-without-wri

#ifndef MOVIE_H
#define MOVIE_H

#include <cairo/cairo.h>
#include <stdint.h>
#include <string>
#include <vector>

extern "C"
{
	#include <x264.h>
	#include <libswscale/swscale.h>
	#include <libavcodec/avcodec.h>
	#include <libavutil/mathematics.h>
	#include <libavformat/avformat.h>
	#include <libavutil/opt.h>
}

class MovieWriter
{
    unsigned int width, height;
    unsigned int iframe;
    int frameRate;

    bool writing=false;

    SwsContext* swsCtx;
    AVOutputFormat* fmt;
    AVStream* stream;
    AVFormatContext* fc;
    AVCodecContext* c;
    AVPacket pkt;

    AVFrame *rgbpic, *yuvpic;

    std::vector<uint8_t> pixels;

    cairo_surface_t* cairo_surface;

public :
    MovieWriter()=default;
    void start_writing(const std::string& filename, const unsigned int width, const unsigned int height, const int frameRate = 25);

    void addFrame(const uint8_t* pixels, bool write_to_console = false);

    void stop_writing();
    ~MovieWriter();
};

class MovieReader
{
	const unsigned int width, height;

	SwsContext* swsCtx;
	AVOutputFormat* fmt;
	AVStream* stream;
	AVFormatContext* fc;
	AVCodecContext* c;
	AVFrame* pFrame;
	AVFrame* pFrameRGB;

	// The index of video stream.
	int ivstream;

    bool stopped_reading = false;

public :

	MovieReader(const std::string& filename, const unsigned int width, const unsigned int height);

	bool getFrame(std::vector<uint8_t>& pixels);
    int getFrameRate() const;

    void stop_reading();

	~MovieReader();	
};

#endif // MOVIE_H

