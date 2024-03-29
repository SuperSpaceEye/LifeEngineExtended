// Adapted from https://stackoverflow.com/questions/34511312/how-to-encode-a-video-from-several-images-generated-in-a-c-program-without-wri

#include "../include/movie.h"

#include <librsvg-2.0/librsvg/rsvg.h>
#include <vector>

using namespace std;

// One-time initialization.
class FFmpegInitialize
{
public :

    FFmpegInitialize()
    {
        // Loads the whole database of available codecs and formats.
        av_register_all();
    }
};

static FFmpegInitialize ffmpegInitialize;

void MovieWriter::start_writing(const string& filename_, const unsigned int width_, const unsigned int height_, const int frameRate_) {
    width = width_;
    height = height_;
    iframe = 0;
    frameRate = frameRate_;

    cairo_surface = cairo_image_surface_create_for_data(
            (unsigned char*)&pixels[0], CAIRO_FORMAT_RGB24, width, height,
            cairo_format_stride_for_width(CAIRO_FORMAT_RGB24, width));

    // Preparing the data concerning the format and codec,
    // in order to write properly the header, frame data and end of file.
    const char* fmtext = "mp4";
    const string filename = filename_ + "." + fmtext;
    fmt = av_guess_format(fmtext, NULL, NULL);
    avformat_alloc_output_context2(&fc, NULL, NULL, filename.c_str());

    // Setting up the codec.
    AVCodec* codec = avcodec_find_encoder_by_name("libx264");
    AVDictionary* opt = NULL;
    av_dict_set(&opt, "preset", "ultrafast", 0);
    av_dict_set(&opt, "crf", "23", 0);
    stream = avformat_new_stream(fc, codec);
    c = stream->codec;
    c->width = width;
    c->height = height;
    c->pix_fmt = AV_PIX_FMT_YUV420P;
    c->time_base = (AVRational){ 1, frameRate };
    c->framerate = (AVRational){frameRate, 1};
    stream->avg_frame_rate = (AVRational){frameRate, 1};

    // Setting up the format, its stream(s),
    // linking with the codec(s) and write the header.
    if (fc->oformat->flags & AVFMT_GLOBALHEADER)
    {
        // Some formats require a global header.
        c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    avcodec_open2(c, codec, &opt);
    av_dict_free(&opt);

    // Once the codec is set up, we need to let the container know
    // which codec are the streams using, in this case the only (video) stream.
    stream->time_base = (AVRational){ 1, frameRate };
    av_dump_format(fc, 0, filename.c_str(), 1);
    avio_open(&fc->pb, filename.c_str(), AVIO_FLAG_WRITE);
    int ret = avformat_write_header(fc, &opt);
    av_dict_free(&opt);

    //the temporary holder for converted yuv data before calling avpicture_fill on yuvpic
    temp_data.resize(3*width*height);

    // Allocating memory for each conversion output YUV frame.
    yuvpic = av_frame_alloc();
    yuvpic->format = AV_PIX_FMT_YUV420P;
    yuvpic->width = width;
    yuvpic->height = height;
    ret = av_frame_get_buffer(yuvpic, 1);

    writing = true;
}

void MovieWriter::addYUVFrame(const uint8_t *pixels) {
    avpicture_fill((AVPicture*)yuvpic, pixels, AV_PIX_FMT_YUV420P, width, height);

    av_init_packet(&pkt);
    pkt.data = nullptr;
    pkt.size = 0;

    // The PTS of the frame are just in a reference unit,
    // unrelated to the format we are using. We set them,
    // for instance, as the corresponding frame number.
    yuvpic->pts = iframe;

    int got_output;
    int ret = avcodec_encode_video2(c, &pkt, yuvpic, &got_output);
    if (got_output)
    {
        // We set the packet PTS and DTS taking in the account our FPS (second argument),
        // and the time base that our selected format uses (third argument).
        av_packet_rescale_ts(&pkt, (AVRational){ 1, frameRate }, stream->time_base);

        iframe++;

        pkt.stream_index = stream->index;

        // Write the encoded frame to the mp4 file.
        av_interleaved_write_frame(fc, &pkt);
        av_packet_unref(&pkt);
    }
}

// https://stackoverflow.com/questions/9465815/rgb-to-yuv420-algorithm-efficiency
void Bitmap2Yuv420p_calc2(uint8_t *destination, const uint8_t *rgb, size_t width, size_t height)
{
    size_t image_size = width * height;
    size_t upos = image_size;
    size_t vpos = upos + upos / 4;
    size_t i = 0;

    for( size_t line = 0; line < height; ++line )
    {
        if( !(line % 2) )
        {
            for( size_t x = 0; x < width; x += 2 )
            {
                uint8_t r = rgb[4 * i + 2];
                uint8_t g = rgb[4 * i + 1];
                uint8_t b = rgb[4 * i + 0];

                destination[i++] = ((66*r + 129*g + 25*b) >> 8) + 16;

                destination[upos++] = ((-38*r + -74*g + 112*b) >> 8) + 128;
                destination[vpos++] = ((112*r + -94*g + -18*b) >> 8) + 128;

                r = rgb[4 * i + 2];
                g = rgb[4 * i + 1];
                b = rgb[4 * i + 0];

                destination[i++] = ((66*r + 129*g + 25*b) >> 8) + 16;
            }
        }
        else
        {
            for( size_t x = 0; x < width; x += 1 )
            {
                uint8_t r = rgb[4 * i + 2];
                uint8_t g = rgb[4 * i + 1];
                uint8_t b = rgb[4 * i + 0];

                destination[i++] = ((66*r + 129*g + 25*b) >> 8) + 16;
            }
        }
    }
}

void MovieWriter::addFrame(const uint8_t *pixels)
{
    Bitmap2Yuv420p_calc2(temp_data.data(), pixels, width, height);
    avpicture_fill((AVPicture*)yuvpic, temp_data.data(), AV_PIX_FMT_YUV420P, width, height);

    av_init_packet(&pkt);
    pkt.data = NULL;
    pkt.size = 0;

    // The PTS of the frame are just in a reference unit,
    // unrelated to the format we are using. We set them,
    // for instance, as the corresponding frame number.
    yuvpic->pts = iframe;

    int got_output;
    int ret = avcodec_encode_video2(c, &pkt, yuvpic, &got_output);
    if (got_output)
    {
        // We set the packet PTS and DTS taking in the account our FPS (second argument),
        // and the time base that our selected format uses (third argument).
        av_packet_rescale_ts(&pkt, (AVRational){ 1, frameRate }, stream->time_base);

        iframe++;

        pkt.stream_index = stream->index;

        // Write the encoded frame to the mp4 file.
        av_interleaved_write_frame(fc, &pkt);
        av_packet_unref(&pkt);
    }
}

void MovieWriter::convert_image(const uint8_t *pixels) {
    for (unsigned int y = 0; y < height; y++)
    {
        for (unsigned int x = 0; x < width; x++)
        {
            // rgbpic->linesize[0] is equal to width.
            rgbpic->data[0][y * rgbpic->linesize[0] + 3 * x + 0] = pixels[y * 4 * width + 4 * x + 2];
            rgbpic->data[0][y * rgbpic->linesize[0] + 3 * x + 1] = pixels[y * 4 * width + 4 * x + 1];
            rgbpic->data[0][y * rgbpic->linesize[0] + 3 * x + 2] = pixels[y * 4 * width + 4 * x + 0];
        }
    }
}

void MovieWriter::stop_writing() {
    // Writing the delayed frames:
    for (int got_output = 1; got_output; )
    {
        int ret = avcodec_encode_video2(c, &pkt, NULL, &got_output);
        if (got_output)
        {
            fflush(stdout);
            av_packet_rescale_ts(&pkt, (AVRational){ 1, frameRate }, stream->time_base);
            pkt.stream_index = stream->index;
            printf("Writing frame %d (size = %d)\n", iframe++, pkt.size);
            av_interleaved_write_frame(fc, &pkt);
            av_packet_unref(&pkt);
        }
    }

    // Writing the end of the file.
    av_write_trailer(fc);

    // Closing the file.
    if (!(fmt->flags & AVFMT_NOFILE))
        avio_closep(&fc->pb);
    avcodec_close(stream->codec);

    // Freeing all the allocated memory:
    temp_data = std::vector<unsigned char>{};
    av_frame_free(&yuvpic);
    avformat_free_context(fc);

    cairo_surface_destroy(cairo_surface);
    writing = false;
}

MovieWriter::~MovieWriter()
{
    if (writing) {stop_writing();}
}