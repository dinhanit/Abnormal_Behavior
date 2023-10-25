function addTask() {
    const taskInput = document.getElementById("taskInput");
    const taskDescription = taskInput.value;
    if (taskDescription) {
        fetch("/add_task/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ "description": taskDescription }),
        })
        .then(() => {
            taskInput.value = "";
            location.reload();
        });
    }
}

function removeTask(index) {
    fetch("/remove_task/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify(index),
    })
    .then(() => {
        location.reload();
    });
}
