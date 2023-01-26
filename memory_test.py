const_array = [];

while (true) {
    for (let i = 0; i < 100000; i++) {
        array.push(i);
    }

    console.log(process.memoryUsage());
}