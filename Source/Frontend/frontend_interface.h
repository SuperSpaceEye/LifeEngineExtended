//
// Created by spaceeye on 03.06.23.
//

#ifndef LIFEENGINEEXTENDED_FRONTEND_INTERFACE_H
#define LIFEENGINEEXTENDED_FRONTEND_INTERFACE_H

#include __FRONTEND_IMPL_RELATIVE_PATH__

//intention here is to abstract frontend (qt) to make transition to other frontends (html, css, idk) easy and to
//simplify adding and modifying various ui elements.

namespace UiFrontend {
    typedef FrontendImpl::Window UiWindow;
}

#endif //LIFEENGINEEXTENDED_FRONTEND_INTERFACE_H
