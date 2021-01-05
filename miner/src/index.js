#!/usr/bin/node
const fs = require('fs');
const execSync = require('child_process').execSync;

// const SSTUBS = 'sstubs-0104.json';
const SSTUBS = 'sstubsLarge-0104.json';

let sstubs;

try {
    sstubs = fs.readFileSync(`${SSTUBS}`, 'utf8');
} catch (error) {
    console.log('Could not open SStuBs file');
    return;
}

try {
    sstubs = JSON.parse(sstubs);
} catch (error) {
    console.log('Could not parse SStuBs file');
    return;
}

let projects = [];

sstubs.forEach(sstub => {
    if (!projects.includes(sstub['projectName'].toLowerCase())) {
        projects.push(sstub['projectName'].toLowerCase());
    }
});

// Preventive cleanup
execSync(`rm -rf tmp`);
execSync(`mkdir tmp`);

fs.writeFileSync(`enriched${SSTUBS}`, `[`);

projects.forEach(project => {
    console.log(`${projects.indexOf(project) + 1}/${projects.length} : ${project}`);

    execSync(`git clone https://github.com/${project.replace('.', '/')} tmp/${project} > /dev/null`);
    
    sstubs.filter(val => val.projectName.toLowerCase() === project).forEach(val => {
        try {
            execSync(`cd tmp/${project}; git checkout ${val.fixCommitSHA1} 2> /dev/null`);
            let index = sstubs.findIndex(cur => cur.fixCommitSHA1 === val.fixCommitSHA1);

            // Get file Lines, Words and Bytes
            let result = execSync(`wc tmp/${project}/${val.bugFilePath}`).toString();
            let match = result.match(/[ ]*([0-9]*)[ ]*([0-9]*)[ ]*([0-9]*).*/);
            sstubs[index]['bugFileTotalLines'] = match[1];
            sstubs[index]['bugFileTotalWords'] = match[2];
            sstubs[index]['bugFileTotalBytes'] = match[3];
            sstubs[index]['bugRelativePos'] = sstubs[index]['bugLineNum'] * 100 / match[1];

            // Get file depth
            sstubs[index]['fileDepthNumber'] = (val.bugFilePath.match(/\//g) || []).length;

            // Get time since previous commit
            let log = execSync(`git -C tmp/${project} log -2 --pretty=format:%aI`).toString();
            let time1 = new Date(log.split('\n')[0])
            let time2 = new Date(log.split('\n')[1])
            sstubs[index]['secSincePreviousCommit'] = (time1 - time2) / 1000;

            fs.appendFileSync(`enriched${SSTUBS}`, `${JSON.stringify(sstubs[index])},`);
        } catch (error) {
            console.error(error);
        }
    });

    // Store project license for later analysis
    // execSync(`mkdir -p tmp/licenses/${project}`);
    // execSync(`cp tmp/${project}/[lL][iI][sCcC][eE][nN][sScC]* tmp/licenses/${project} || true`);

    execSync(`rm -rf tmp/${project}`);
});

fs.appendFileSync(`enriched${SSTUBS}`, `]`);
