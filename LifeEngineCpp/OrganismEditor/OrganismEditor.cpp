//
// Created by spaceeye on 25.06.22.
//

#include "OrganismEditor.h"

void OrganismEditor::init(int width, int height, Ui::MainWindow *parent_ui) {
    _ui.setupUi(this);
    _parent_ui = parent_ui;

    editor_width = width;
    editor_height = height;
}

void OrganismEditor::closeEvent(QCloseEvent * event) {
    _parent_ui->tb_open_organism_editor->setChecked(false);
    QWidget::closeEvent(event);
}

