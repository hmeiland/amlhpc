cat > license.ini << EOF
[tNavigator_license_settings]
url=tnav://xxxxxxxxxxxxxxxxxxxxxxx
login=xxxxxxxxxxxxxx
pass=xxxxxxxxxxxx
EOF

ls SpeedTestModel

/opt/tNavigator-Linux-64/tNavigator-con --license-settings=./license.ini --no-gui ./SpeedTestModel/MODEL.DAT

cp -R SpeedTestModel/RESULTS ./outputs/
