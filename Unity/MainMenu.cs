using UnityEngine;
//using UnityEngine.SceneManagement;
public class MainMenu : MonoBehaviour
{
    private int selectedSlot;

    public void NewGame(int slot)
    {
        if (slot < 1 || slot > 5)
        {
            return;
        }

//SceneManagement.LoadScene("Game");

    }

    public void ContinueGame(int slot)
    {
        if (slot < 1 || slot > 5)
        {
            return;
        }

        if (SavesExists(slot))
        {
            return;
        }
        //Загрузить сохранение
    }

    public void DeleteSave(int slot)
    {
        if (slot < 1 || slot > 5)
        {
            return;
        }

        selectedSlot = slot;
    }

    public void ConfirmDelete()
    {
        
    }

    public void CancelAction()

    {

    }

    public void OpenSettings()
    {
        
    }

    public void CloseSetting()
    {
        
    }

    public void QuitGame()
    {

    }

    public void ConfirmQuit()
    {
        Application.Quit();
    }

    public void CancelQuit()
    {
        
    }
}