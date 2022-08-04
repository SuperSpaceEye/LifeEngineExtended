if .ui files are changed, run before building.
uic -g cpp Source/UiFiles/mainwindow.ui -o Source/MainWindow/WindowUI.h
uic -g cpp Source/UiFiles/statistics.ui -o Source/Statistics/StatisticsUI.h
uic -g cpp Source/UiFiles/editor.ui -o Source/OrganismEditor/EditorUI.h
uic -g cpp Source/UiFiles/info.ui -o Source/InfoWindow/InfoWindowUI.h
uic -g cpp Source/UiFiles/recorder.ui -o Source/Recorder/RecorderWindowUI.h
uic -g cpp Source/UiFiles/worldevents.ui -o Source/WorldEvents/WorldEventsUI.h
uic -g cpp Source/UiFiles/ChangeValueEventNodeWidget.ui -o Source/WorldEvents/ChangeValueEventNodeWidgetUI.h
uic -g cpp Source/UiFiles/ConditionalEventNodeWidget.ui -o Source/WorldEvents/ConditionalEventNodeWidgetUI.h


uic -g cpp Source/UiFiles/mainwindow.ui -o Source/MainWindow/WindowUI.h ; uic -g cpp Source/UiFiles/statistics.ui -o Source/Statistics/StatisticsUI.h ; uic -g cpp Source/UiFiles/editor.ui -o Source/OrganismEditor/EditorUI.h ; uic -g cpp Source/UiFiles/info.ui -o Source/InfoWindow/InfoWindowUI.h ; uic -g cpp Source/UiFiles/recorder.ui -o Source/Recorder/RecorderWindowUI.h ; uic -g cpp Source/UiFiles/worldevents.ui -o Source/WorldEvents/WorldEventsUI.h ; uic -g cpp Source/UiFiles/ChangeValueEventNodeWidget.ui -o Source/WorldEvents/ChangeValueEventNodeWidgetUI.h; uic -g cpp Source/UiFiles/ConditionalEventNodeWidget.ui -o Source/WorldEvents/ConditionalEventNodeWidgetUI.h