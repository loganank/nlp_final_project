import React, { useEffect, useState } from 'react'
import {styles} from './style'

function App() {

  const [inputValue, setInputValue] = useState('')
  const [classifier, setClassifier] = useState('')

  useEffect(() => {
    // fetch("/api").then(
    //   res => res.json()
    // ).then(
    //   data => {
    //     setData(data)
    //   }
    // )
  }, [])


  const handleClick = () => {
    console.log(inputValue)
    fetch('/evaluateText', {
      method: 'POST',
      headers: {
        'Content-Type' : 'application/json'
      },
      body: JSON.stringify(inputValue)
    }).then(
        res => res.json() // parse response as JSON
    ).then(
        data => setClassifier(data['classifier']) // pass parsed JSON to createModelOutput function
    ).catch(
        error => console.error(error) // handle any errors
    )
  }

  return (
    <div style={styles.container}>

      <p style={styles.text}>The text that you submit in the following text box is given to a machine learning model that guesses whether the text is suicidal, depressed, or neither.</p>

      <p style={styles.text}>This is for demonstration purposes only, and the model is prone to making many mistakes.</p>

      <textarea style={styles.input} value={inputValue} onChange={e => setInputValue(e.target.value)}></textarea>

      <button style={styles.button} onClick={handleClick}>Submit</button>

      {(classifier === '') ? (
        <p style={styles.text}>Enter text to be evaluated.</p>
      ): (
          <p style={styles.text}>The text is {classifier}.</p>
      )}

    </div>
  )
}

export default App