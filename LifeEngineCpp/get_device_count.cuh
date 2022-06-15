//
// Created by SpaceEye on 12.06.22.
//

#ifndef THELIFEENGINECPP_GET_DEVICE_COUNT_CUH
#define THELIFEENGINECPP_GET_DEVICE_COUNT_CUH

#if __WIN32
    #pragma once
    #ifdef GET_DEVICE_COUNT_EXPORTS
        #define GET_DEVICE_COUNT_API __declspec(dllexport)
    #else
        #define GET_DEVICE_COUNT_API __declspec(dllimport)
    #endif
#else
    #define GET_DEVICE_COUNT_API
#endif

GET_DEVICE_COUNT_API int get_device_count();
int get_device_count();

#endif //THELIFEENGINECPP_GET_DEVICE_COUNT_CUH
