fetch("/get", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `msg=${encodeURIComponent(userText)}`
})
.then(res => res.json())
.then(data => {
    document.getElementById("chatbox").innerHTML += `<div class='bot'>Bot: ${data.response}</div>`;
    document.getElementById("chatbox").scrollTop = chatbox.scrollHeight;
});
