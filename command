if .ui files are changed, run before building.
uic -g cpp LifeEngineCpp/UiFiles/mainwindow.ui -o LifeEngineCpp/MainWindow/WindowUI.h
uic -g cpp LifeEngineCpp/UiFiles/statistics.ui -o LifeEngineCpp/Statistics/StatisticsUI.h
uic -g cpp LifeEngineCpp/UiFiles/editor.ui -o LifeEngineCpp/OrganismEditor/EditorUI.h


uic -g cpp LifeEngineCpp/UiFiles/mainwindow.ui -o LifeEngineCpp/MainWindow/WindowUI.h ; uic -g cpp LifeEngineCpp/UiFiles/statistics.ui -o LifeEngineCpp/Statistics/StatisticsUI.h ; uic -g cpp LifeEngineCpp/UiFiles/editor.ui -o LifeEngineCpp/OrganismEditor/EditorUI.h