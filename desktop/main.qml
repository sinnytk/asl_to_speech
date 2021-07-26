import QtQuick 2.14
import QtQuick.Controls 2.14
import QtQuick.Layouts 1.14
import QtMultimedia 5.15

ApplicationWindow {
    visible: true
    title: "ASL To Speech - FYP - Tarun, Bahawal, Hira"
    flags: Qt.Window
    width: 1200
    height: 800

    ColumnLayout {
        id: containerLayout
        spacing: 0
        anchors.centerIn: parent
        RowLayout {
            id: contentLayout
            spacing: 5
            Rectangle {
                id: streamContainer
                Layout.alignment: Qt.AlignCenter
                Layout.preferredWidth: 1200 / 2 - 5
                Layout.preferredHeight: 650

                VideoOutput {
                    visible: GUIBackend.is_feed_open
                    source: AbstractStreamAdapter
                    anchors.fill: parent
                    
                }
                
            }
            Rectangle {
                id: annotationsContainer
                Layout.alignment: Qt.AlignCenter
                Layout.preferredWidth: 1200 / 2 - 5
                Layout.preferredHeight: 650
                Text {
                    text: AbstractStreamAdapter.annotation
                    font.family: "Helvetica"
                    font.pointSize: 24
                    color: "red"
                }
            }
        }

        Text {
            id: statusText
            text: GUIBackend.status
            Layout.alignment: Qt.AlignCenter

        }

        Rectangle {
            id: buttonContainer
            Layout.alignment: Qt.AlignCenter
            Layout.preferredWidth: 1200
            Layout.preferredHeight: 150

            Button {
                id: startBtn
                text: "Start!"
                anchors.centerIn: parent
                background: Rectangle {
                    implicitWidth: 300
                    implicitHeight: 50
                    opacity: enabled ? 1 : 0.3
                    color: startBtn.down || startBtn.hovered ? "#d0d0d0" : "#e0e0e0"
                }
                onClicked: GUIBackend.start_webcam_feed()

            }
        }
    }
}