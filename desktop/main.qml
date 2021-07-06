import QtQuick
import QtQuick.Controls
import QtQuick.Layouts


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
                color: "lightblue"
                Layout.preferredWidth: 1200 / 2 - 5
                Layout.preferredHeight: 650
                
            }
            Rectangle {
                id: annotationsContainer
                Layout.alignment: Qt.AlignCenter
                color: "lightpink"
                Layout.preferredWidth: 1200 / 2 - 5
                Layout.preferredHeight: 650
                
            }
        }

        Text {
            id: statusText
            text: GUIBackend.status
            
            anchors.horizontalCenter: parent.horizontalCenter
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