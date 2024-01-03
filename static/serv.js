// script.js

document.addEventListener("DOMContentLoaded", function() {
 // Check if the element with ID "serv" exists
 var servicesLink = document.getElementById('serv');

 if (servicesLink) {
     // Attach a click event listener to the element
     servicesLink.addEventListener('click', function(event) {
         // Prevent the default behavior of the link
         event.preventDefault();

         // Display an alert message
         alert("Please login to access services");
     });
 }
});


